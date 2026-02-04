"""
Skills Promotion Service.

Automatically promotes successful fixes to reusable skills when criteria met:
- 3+ episodes with similar fix (similarity > 0.85)
- Fix success rate > 90%
- Fix contains executable code or clear procedure

Security:
- Customer namespace isolation
- Input validation
- LLM output validation for generated names/docstrings
"""

import logging
import re

from src.ingestion.embedding import EmbeddingService
from src.kyrodb.router import KyroDBRouter
from src.models.episode import Episode
from src.models.skill import Skill

logger = logging.getLogger(__name__)


class SkillPromotionService:
    """
    Automatically promotes successful fixes to reusable skills.

    Security:
    - All operations customer-scoped
    - Similarity thresholds prevent false promotions
    - LLM-generated names validated
    """

    # Promotion criteria
    MIN_EPISODES_FOR_PROMOTION = 3
    MIN_SIMILARITY_THRESHOLD = 0.85
    MIN_SUCCESS_RATE = 0.9

    # Code detection patterns
    CODE_PATTERNS = [
        r"```[\w]*\n",  # Code blocks
        r"def\s+\w+",  # Python functions
        r"function\s+\w+",  # JavaScript functions
        r"class\s+\w+",  # Class definitions
        r"kubectl\s+",  # Kubectl commands
        r"docker\s+",  # Docker commands
        r"git\s+",  # Git commands
        r"npm\s+",  # NPM commands
        r"pip\s+",  # Pip commands
    ]

    def __init__(
        self,
        kyrodb_router: KyroDBRouter,
        embedding_service: EmbeddingService,
    ):
        """
        Initialize promotion service.

        Args:
            kyrodb_router: KyroDB router for storage
            embedding_service: Embedding service for similarity
        """
        self.kyrodb_router = kyrodb_router
        self.embedding_service = embedding_service

    async def check_and_promote(
        self,
        episode_id: int,
        customer_id: str,
    ) -> Skill | None:
        """
        Check if episode should be promoted to skill.

        Security:
        - Customer ID validated
        - Episode must meet strict criteria
        - Similar episodes verified

        Args:
            episode_id: Episode to check for promotion
            customer_id: Customer namespace

        Returns:
            Skill if promoted, None otherwise
        """
        if not customer_id:
            raise ValueError("customer_id is required for skill promotion")

        try:
            # Step 1: Fetch episode
            episode = await self._fetch_episode(episode_id, customer_id)
            if not episode:
                logger.debug(f"Episode {episode_id} not found for promotion check")
                return None

            # Step 2: Check if fix is substantial
            if not self._has_substantial_fix(episode):
                logger.debug(f"Episode {episode_id} fix not substantial enough for promotion")
                return None

            # Step 3: Check success rate
            if episode.usage_stats.fix_success_rate < self.MIN_SUCCESS_RATE:
                logger.debug(
                    f"Episode {episode_id} success rate too low: "
                    f"{episode.usage_stats.fix_success_rate:.2f} "
                    f"(threshold: {self.MIN_SUCCESS_RATE})"
                )
                return None

            # Step 4: Find similar episodes
            similar_episodes = await self._find_similar_episodes(episode, customer_id)

            if len(similar_episodes) < self.MIN_EPISODES_FOR_PROMOTION:
                logger.debug(
                    f"Episode {episode_id} has only {len(similar_episodes)} similar episodes "
                    f"(need {self.MIN_EPISODES_FOR_PROMOTION})"
                )
                return None

            # Step 5: Check similar episodes' success rate
            avg_success_rate = sum(e.usage_stats.fix_success_rate for e in similar_episodes) / len(
                similar_episodes
            )

            if avg_success_rate < self.MIN_SUCCESS_RATE:
                logger.debug(f"Similar episodes avg success rate too low: {avg_success_rate:.2f}")
                return None

            # Step 6: Promote to skill
            skill = await self._create_skill(episode, similar_episodes, customer_id)

            logger.info(
                f"Promoted episode {episode_id} to skill {skill.skill_id} "
                f"(from {len(similar_episodes)} similar episodes, "
                f"avg success rate: {avg_success_rate:.2f})"
            )

            return skill

        except Exception as e:
            logger.error(
                f"Skill promotion check failed for episode {episode_id}: {e}",
                exc_info=True,
            )
            return None

    async def _fetch_episode(self, episode_id: int, customer_id: str) -> Episode | None:
        """
        Fetch episode from KyroDB.

        Args:
            episode_id: Episode ID
            customer_id: Customer namespace

        Returns:
            Episode if found, None otherwise
        """
        from src.kyrodb.router import get_namespaced_collection

        collection = "failures"
        namespaced_collection = get_namespaced_collection(customer_id, collection)

        try:
            response = await self.kyrodb_router.text_client.query(
                doc_id=episode_id,
                namespace=namespaced_collection,
                include_embedding=False,
            )

            if not response.found:
                return None

            # Deserialize episode
            raw_metadata = getattr(response, "metadata", None)
            metadata: dict[str, str] = {}
            if raw_metadata:
                if isinstance(raw_metadata, dict):
                    metadata = raw_metadata
                else:
                    try:
                        if hasattr(raw_metadata, "items"):
                            metadata = dict(raw_metadata.items())
                        else:
                            metadata = dict(raw_metadata)
                    except (TypeError, ValueError, AttributeError):
                        metadata = {}

            episode = Episode.from_metadata_dict(episode_id, metadata)
            return episode

        except Exception as e:
            logger.error(f"Failed to fetch episode {episode_id}: {e}")
            return None

    def _has_substantial_fix(self, episode: Episode) -> bool:
        """
        Check if reflection contains a substantial fix worth promoting.

        A substantial fix is:
        - Has code blocks OR
        - Has numbered steps AND is detailed (>100 chars)

        Args:
            episode: Episode to check

        Returns:
            bool: True if fix is substantial
        """
        if not episode.reflection:
            return False

        resolution = episode.reflection.resolution_strategy

        # Check for code
        has_code = any(re.search(pattern, resolution) for pattern in self.CODE_PATTERNS)

        # Check for procedural steps
        step_patterns = [
            r"\n\d+\.",  # Numbered list (1., 2., etc.)
            r"\nStep \d+:",  # "Step 1:", "Step 2:", etc.
            r"\n- ",  # Bullet points
        ]
        has_steps = any(re.search(pattern, resolution) for pattern in step_patterns)

        # Check detail level
        has_detail = len(resolution) > 100

        return has_code or (has_steps and has_detail)

    async def _find_similar_episodes(
        self,
        episode: Episode,
        customer_id: str,
    ) -> list[Episode]:
        """
        Find similar episodes with same/similar fix.

        Security:
        - Customer namespace isolation
        - Similarity threshold enforced

        Args:
            episode: Base episode
            customer_id: Customer namespace

        Returns:
            list: Similar episodes that meet criteria
        """
        from src.kyrodb.router import get_namespaced_collection

        collection = "failures"
        namespaced_collection = get_namespaced_collection(customer_id, collection)
        episode_reflection = episode.reflection
        if episode_reflection is None:
            return []

        try:
            # Generate embedding for episode goal
            query_text = f"{episode.create_data.goal}\n\n{episode.create_data.error_trace[:500]}"
            query_embedding = await self.embedding_service.embed_text_async(query_text)

            # Search for similar episodes
            response = await self.kyrodb_router.text_client.search(
                query_embedding=query_embedding,
                k=20,  # Get more candidates
                namespace=namespaced_collection,
                min_score=self.MIN_SIMILARITY_THRESHOLD,
                include_embeddings=False,
            )

            similar_episodes = []
            for result in response.results:
                # Skip self
                if result.doc_id == episode.episode_id:
                    continue

                try:
                    similar_ep = Episode.from_metadata_dict(result.doc_id, dict(result.metadata))

                    # Must have reflection
                    if not similar_ep.reflection:
                        continue

                    # Must have been applied at least once
                    if similar_ep.usage_stats.fix_applied_count == 0:
                        continue

                    # Must have success rate tracked
                    total_validations = (
                        similar_ep.usage_stats.fix_success_count
                        + similar_ep.usage_stats.fix_failure_count
                    )
                    if total_validations == 0:
                        continue

                    # Check resolution similarity (text-based)
                    similar_reflection = similar_ep.reflection
                    if similar_reflection is None:
                        continue
                    if self._resolutions_similar(
                        episode_reflection.resolution_strategy,
                        similar_reflection.resolution_strategy,
                    ):
                        similar_episodes.append(similar_ep)

                except Exception as e:
                    logger.warning(f"Failed to process similar episode {result.doc_id}: {e}")
                    continue

            logger.debug(f"Found {len(similar_episodes)} similar episodes for {episode.episode_id}")

            return similar_episodes

        except Exception as e:
            logger.error(f"Similar episodes search failed: {e}", exc_info=True)
            return []

    def _resolutions_similar(self, res1: str, res2: str) -> bool:
        """
        Check if two resolutions are similar using simple text matching.

        This is a heuristic check. Future: Use embedding similarity.

        Args:
            res1: First resolution
            res2: Second resolution

        Returns:
            bool: True if resolutions appear similar
        """
        # Normalize
        res1_normalized = res1.lower().strip()
        res2_normalized = res2.lower().strip()

        # Extract key terms (words > 5 chars)
        words1 = {word for word in re.findall(r"\w+", res1_normalized) if len(word) > 5}
        words2 = {word for word in re.findall(r"\w+", res2_normalized) if len(word) > 5}

        if not words1 or not words2:
            return False

        # Jaccard similarity
        intersection = len(words1 & words2)
        union = len(words1 | words2)

        if union == 0:
            return False

        jaccard = intersection / union

        # Consider similar if Jaccard > 0.5
        return jaccard > 0.5

    async def _create_skill(
        self,
        primary_episode: Episode,
        similar_episodes: list[Episode],
        customer_id: str,
    ) -> Skill:
        """
        Create skill from episode cluster.

        Security:
        - Generated name validated
        - Customer ID enforced
        - Source episodes tracked

        Args:
            primary_episode: Main episode to promote
            similar_episodes: Similar episodes
            customer_id: Customer namespace

        Returns:
            Skill: Created and stored skill
        """
        # Combine all episodes for analysis
        all_episodes = [primary_episode] + similar_episodes
        source_episode_ids = [ep.episode_id for ep in all_episodes]

        # Extract skill metadata
        skill_name = self._generate_skill_name(primary_episode)
        skill_docstring = self._generate_skill_docstring(primary_episode, all_episodes)

        # Extract code or procedure
        primary_reflection = primary_episode.reflection
        if primary_reflection is None:
            raise ValueError("Primary episode reflection is required for skill creation")
        resolution = primary_reflection.resolution_strategy
        code, procedure, language = self._extract_code_or_procedure(resolution)

        # Extract categorization
        tags = self._extract_tags(all_episodes)
        tools = list(
            {tool for ep in all_episodes for tool in ep.create_data.tool_chain[:2]}  # Top 2 tools
        )

        # Create skill
        from src.storage.database import get_customer_db

        db = await get_customer_db()
        skill_id = await db.allocate_doc_id()
        skill = Skill(
            skill_id=skill_id,
            customer_id=customer_id,
            name=skill_name,
            docstring=skill_docstring,
            code=code,
            procedure=procedure,
            language=language,
            source_episodes=source_episode_ids,
            tags=tags,
            error_class=primary_episode.create_data.error_class.value,
            tools=tools,
        )

        # Generate embedding for skill
        embedding_parts: list[str] = [
            f"name: {skill.name}",
            f"docstring: {skill.docstring}",
            f"error_class: {skill.error_class}",
        ]

        if skill.tools:
            embedding_parts.append(f"tools: {', '.join(skill.tools[:10])}")
        if skill.tags:
            embedding_parts.append(f"tags: {', '.join(skill.tags[:20])}")

        # Include representative episode context so skills are discoverable by the
        # same goal/error phrasing developers will search with.
        embedding_parts.append(f"example_goal: {primary_episode.create_data.goal}")
        embedding_parts.append(f"example_error: {primary_episode.create_data.error_trace[:500]}")

        if primary_episode.reflection:
            embedding_parts.append(f"root_cause: {primary_episode.reflection.root_cause}")
            embedding_parts.append(
                f"resolution_strategy: {primary_episode.reflection.resolution_strategy[:300]}"
            )

        embedding_text = "\n\n".join(embedding_parts)
        skill_embedding = await self.embedding_service.embed_text_async(embedding_text)

        # Store in KyroDB
        success = await self.kyrodb_router.insert_skill(skill, skill_embedding)

        if not success:
            raise RuntimeError(f"Failed to store skill {skill_id} in KyroDB")

        logger.info(
            f"Created skill {skill_id}: {skill_name} (from {len(source_episode_ids)} episodes)"
        )

        return skill

    def _generate_skill_name(self, episode: Episode) -> str:
        """
        Generate skill name from episode.

        Security: Validates and sanitizes generated name.

        Args:
            episode: Episode to generate name from

        Returns:
            str: Sanitized skill name
        """
        # Extract key terms from goal and error
        goal = episode.create_data.goal.lower()
        tool = episode.create_data.tool_chain[0] if episode.create_data.tool_chain else "unknown"

        # Simple name generation (future: use LLM)
        # Format: {action}_{tool}_{error_type}
        # Example: "fix_kubectl_image_pull_error"

        # Extract action verb
        action_verbs = ["fix", "resolve", "configure", "deploy", "update", "install"]
        action = next((verb for verb in action_verbs if verb in goal), "fix")

        # Sanitize tool name
        tool_clean = re.sub(r"[^\w]", "_", tool.lower())

        # Extract error type
        error_type = episode.create_data.error_class.value.replace("_error", "")

        name = f"{action}_{tool_clean}_{error_type}"

        # Validate length
        if len(name) > 200:
            name = name[:200]

        # Sanitize
        name = re.sub(r"[^\w_]", "", name)
        name = re.sub(r"_+", "_", name)

        return name

    def _generate_skill_docstring(
        self, primary_episode: Episode, all_episodes: list[Episode]
    ) -> str:
        """
        Generate skill docstring.

        Future: Use LLM to generate better descriptions.

        Args:
            primary_episode: Main episode
            all_episodes: All similar episodes

        Returns:
            str: Sanitized docstring
        """
        # For now, use the root cause from reflection
        if primary_episode.reflection:
            root_cause = primary_episode.reflection.root_cause

            docstring = (
                f"Fixes {primary_episode.create_data.error_class.value.replace('_', ' ')} "
                f"in {primary_episode.create_data.tool_chain[0]}. "
                f"Root cause: {root_cause[:100]}... "
                f"Promoted from {len(all_episodes)} successful resolutions."
            )

            # Sanitize
            docstring = re.sub(r"\s+", " ", docstring).strip()
            if len(docstring) > 2000:
                docstring = docstring[:1997] + "..."

            return docstring

        return "Skill promoted from successful episode resolutions."

    def _extract_code_or_procedure(
        self, resolution: str
    ) -> tuple[str | None, str | None, str | None]:
        """
        Extract code, procedure, and language from resolution.

        Args:
            resolution: Resolution strategy text

        Returns:
            tuple: (code, procedure, language)
        """
        # Try to extract code block
        code_block_pattern = r"```([\w]*)\n(.*?)```"
        match = re.search(code_block_pattern, resolution, re.DOTALL)

        if match:
            language = match.group(1) or "bash"
            code = match.group(2).strip()
            return code, None, language

        # Check if it's a procedural fix
        has_steps = any(pattern in resolution for pattern in ["\n1.", "\n2.", "Step 1:", "Step 2:"])

        if has_steps:
            # Extract procedure
            procedure = resolution.strip()
            if len(procedure) > 5000:
                procedure = procedure[:5000]
            return None, procedure, None

        # Default: treat as procedure
        return None, resolution.strip()[:5000], None

    def _extract_tags(self, episodes: list[Episode]) -> list[str]:
        """
        Extract common tags from episodes.

        Args:
            episodes: List of episodes

        Returns:
            list: Common tags
        """
        tag_counts: dict[str, int] = {}

        for episode in episodes:
            for tag in episode.create_data.tags:
                tag_counts[tag] = tag_counts.get(tag, 0) + 1

        # Return tags that appear in at least 50% of episodes
        threshold = len(episodes) * 0.5
        common_tags = [tag for tag, count in tag_counts.items() if count >= threshold]

        return common_tags[:20]  # Max 20 tags
