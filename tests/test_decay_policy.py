"""
Unit tests for memory decay policy.

Tests archival, deletion, and permanent protection logic.
"""

import pytest
from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock, MagicMock

from src.hygiene.decay import MemoryDecayPolicy
from src.models.episode import Episode, EpisodeCreate, ErrorClass, UsageStats


class TestMemoryDecayPolicy:
    """Test suite for MemoryDecayPolicy."""
    
    @pytest.fixture
    def mock_kyrodb_router(self):
        """Mock KyroDB router."""
        return AsyncMock()
    
    @pytest.fixture
    def decay_policy(self, mock_kyrodb_router):
        """Create decay policy instance."""
        return MemoryDecayPolicy(
            kyrodb_router=mock_kyrodb_router,
            archive_age_days=180,
            delete_unused_days=90
        )
    
    def test_initialization(self, decay_policy):
        """Test policy initialization."""
        assert decay_policy.archive_age_days == 180
        assert decay_policy.delete_unused_days == 90
    
    def test_initialization_validation(self, mock_kyrodb_router):
        """Test validation of age parameters."""
        with pytest.raises(ValueError, match="archive_age_days must be >= 30"):
            MemoryDecayPolicy(mock_kyrodb_router, archive_age_days=20)
        
        with pytest.raises(ValueError, match="delete_unused_days must be >= 30"):
            MemoryDecayPolicy(mock_kyrodb_router, delete_unused_days=20)
    
    def create_episode(
        self,
        episode_id: int,
        created_at: datetime,
        error_class: ErrorClass = ErrorClass.AUTHENTICATION_ERROR,  # Use actual enum value
        retrieval_count: int = 0,
        fix_success_count: int = 0,
        fix_success_rate: float = 0.0,
        metadata: dict = None,
        tags: list[str] = None
    ) -> Episode:
        """Helper to create test episode."""
        return Episode(
            episode_id=episode_id,
            text_embedding=[0.5] * 384,
            image_embedding=None,
            create_data=EpisodeCreate(
                goal="Test goal",
                error_trace="Test error",
                error_class=error_class,
                metadata=metadata or {},
                tags=tags or []
            ),
            reflection=None,
            usage_stats=UsageStats(
                total_retrievals=retrieval_count,
                fix_success_count=fix_success_count,
                fix_success_rate=fix_success_rate
            ),
            created_at=created_at,
            customer_id="test_customer"
        )
    
    def test_is_permanent_critical_error(self, decay_policy):
        """Test permanent protection for critical error classes."""
        now = datetime.now(timezone.utc)
        
        # Configuration error (in PERMANENT_ERROR_CLASSES)
        episode = self.create_episode(
            episode_id=1,
            created_at=now,
            error_class=ErrorClass.CONFIGURATION_ERROR
        )
        
        assert decay_policy._is_permanent(episode) is True
    
    def test_is_permanent_manual_flag(self, decay_policy):
        """Test permanent protection via manual flag."""
        now = datetime.now(timezone.utc)
        
        episode = self.create_episode(
            episode_id=1,
            created_at=now,
            metadata={"permanent": "true"}
        )
        
        assert decay_policy._is_permanent(episode) is True
    
    def test_is_permanent_high_performing_skill(self, decay_policy):
        """Test permanent protection for high-performing skills."""
        now = datetime.now(timezone.utc)
        
        episode = self.create_episode(
            episode_id=1,
            created_at=now,
            fix_success_count=10,
            fix_success_rate=0.8
        )
        
        assert decay_policy._is_permanent(episode) is True
    
    def test_is_permanent_critical_tags(self, decay_policy):
        """Test permanent protection via critical tags."""
        now = datetime.now(timezone.utc)
        
        episode = self.create_episode(
            episode_id=1,
            created_at=now,
            tags=["data_loss", "production"]
        )
        
        assert decay_policy._is_permanent(episode) is True
    
    def test_is_not_permanent(self, decay_policy):
        """Test that normal episodes are not permanent."""
        now = datetime.now(timezone.utc)
        
        episode = self.create_episode(
            episode_id=1,
            created_at=now,
            error_class=ErrorClass.RUNTIME_ERROR,
            fix_success_count=2,
            fix_success_rate=0.5
        )
        
        assert decay_policy._is_permanent(episode) is False
    
    @pytest.mark.asyncio
    async def test_apply_decay_policy_archives_old(self, decay_policy, mock_kyrodb_router):
        """Test that old episodes are archived."""
        now = datetime.now(timezone.utc)
        old_date = now - timedelta(days=200)
        
        old_episode = self.create_episode(
            episode_id=1,
            created_at=old_date,
            retrieval_count=5  # Has been used
        )
        
        decay_policy._fetch_all_episodes = AsyncMock(return_value=[old_episode])
        
        stats = await decay_policy.apply_decay_policy("test_customer", dry_run=False)
        
        assert stats["archived"] == 1
        assert stats["deleted"] == 0
        assert stats["protected"] == 0
    
    @pytest.mark.asyncio
    async def test_apply_decay_policy_deletes_unused(self, decay_policy, mock_kyrodb_router):
        """Test that unused old episodes are deleted."""
        now = datetime.now(timezone.utc)
        unused_date = now - timedelta(days=100)
        
        unused_episode = self.create_episode(
            episode_id=1,
            created_at=unused_date,
            retrieval_count=0  # Never used
        )
        
        decay_policy._fetch_all_episodes = AsyncMock(return_value=[unused_episode])
        
        stats = await decay_policy.apply_decay_policy("test_customer", dry_run=False)
        
        assert stats["archived"] == 0
        assert stats["deleted"] == 1
        assert stats["protected"] == 0
    
    @pytest.mark.asyncio
    async def test_apply_decay_policy_protects_critical(self, decay_policy, mock_kyrodb_router):
        """Test that critical episodes are protected."""
        now = datetime.now(timezone.utc)
        old_date = now - timedelta(days=300)  # Very old
        
        critical_episode = self.create_episode(
            episode_id=1,
            created_at=old_date,
            error_class=ErrorClass.CONFIGURATION_ERROR,  # Critical
            retrieval_count=0  # Unused
        )
        
        decay_policy._fetch_all_episodes = AsyncMock(return_value=[critical_episode])
        
        stats = await decay_policy.apply_decay_policy("test_customer", dry_run=False)
        
        assert stats["archived"] == 0
        assert stats["deleted"] == 0
        assert stats["protected"] == 1
    
    @pytest.mark.asyncio
    async def test_apply_decay_policy_dry_run(self, decay_policy, mock_kyrodb_router):
        """Test dry-run mode."""
        now = datetime.now(timezone.utc)
        old_date = now - timedelta(days=200)
        
        old_episode = self.create_episode(
            episode_id=1,
            created_at=old_date
        )
        
        decay_policy._fetch_all_episodes = AsyncMock(return_value=[old_episode])
        
        stats = await decay_policy.apply_decay_policy("test_customer", dry_run=True)
        
        # Should report what would be done, but not actually do it
        assert stats["archived"] == 1
        # Verify no actual operations were called
        assert not decay_policy._archive_episode.called
    
    @pytest.mark.asyncio
    async def test_apply_decay_policy_customer_validation(self, decay_policy):
        """Test customer ID validation."""
        with pytest.raises(ValueError, match="Invalid customer_id"):
            await decay_policy.apply_decay_policy("")
        
        with pytest.raises(ValueError, match="Invalid customer_id"):
            await decay_policy.apply_decay_policy("a" * 101)
    
    @pytest.mark.asyncio
    async def test_apply_decay_policy_mixed_episodes(self, decay_policy, mock_kyrodb_router):
        """Test policy with mixed episode types."""
        now = datetime.now(timezone.utc)
        
        episodes = [
            # Archive (old, used)
            self.create_episode(1, now - timedelta(days=200), retrieval_count=5),
            # Delete (old, unused)
            self.create_episode(2, now - timedelta(days=100), retrieval_count=0),
            # Protect (critical)
            self.create_episode(
                3,
                now - timedelta(days=300),
                error_class=ErrorClass.CONFIGURATION_ERROR
            ),
            # Keep (recent)
            self.create_episode(4, now - timedelta(days=30)),
        ]
        
        decay_policy._fetch_all_episodes = AsyncMock(return_value=episodes)
        
        stats = await decay_policy.apply_decay_policy("test_customer", dry_run=False)
        
        assert stats["total"] == 4
        assert stats["archived"] == 1
        assert stats["deleted"] == 1
        assert stats["protected"] == 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
