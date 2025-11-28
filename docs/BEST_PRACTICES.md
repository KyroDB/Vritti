# Best Practices

Maximize the value of EpisodicMemory with these integration patterns.

## When to Capture

**DO Capture**:
- Unhandled exceptions
- Tool execution failures (non-zero exit codes)
- Logic failures (agent loop exceeded)
- User-reported failures (feedback)

**DO NOT Capture**:
- Transient network glitches (unless persistent)
- User cancellation
- Successful operations (unless implementing success tracking)

## Search Strategies

### Context is King
Include as much context as possible in your search queries.
- **Bad**: "deployment failed"
- **Good**: "kubectl apply failed with ImagePullBackOff in production"

### Use Preconditions
When searching, provide the `current_state` to enable precondition matching.

```json
"current_state": {
  "os": "linux",
  "cluster_version": "1.24",
  "permissions": "read-only"
}
```

**About Preconditions**:
- **Optional but recommended** for better filtering
- **Available fields**: Include any relevant state (os, cluster_version, permissions, environment, tool_versions, etc.)
- **Matching behavior**: System performs partial matching - episodes are ranked higher if their recorded preconditions are compatible with your current state
- **See [API Reference](API_GUIDE.md#search-for-solutions) for complete list of supported fields**

This helps the system filter out irrelevant episodes (e.g., Windows fixes for Linux problems).

## Performance Optimization

- **Async Capture**: Don't block your agent loop on capture. Fire and forget.

- **Cache Gating**: Cache gating decisions for identical actions for 5-10 minutes.
  - **"Identical actions" = same event type + same normalized action parameters**
  - **Comparison method**: Use deterministic hash of `(goal, proposed_action, tool)`
  - **Rationale**: 
    - Prevents API quota exhaustion
    - Reduces log/alert noise  
    - Improves UX by avoiding repeated identical responses
  - **Configurable window**: Adjust based on your use case:
    - **Shorten (1-2 min)**: High-priority events, user-requested refresh
    - **Extend (15-30 min)**: Batch processing, low-priority background tasks
  - **Bypass gating**: For critical real-time checks or user-initiated actions
  - **Example**:
    - **Gated**: User attempts `rm -rf /data` twice in 3 minutes → serve cached "block" response
    - **Not gated**: 15 minutes later, same user retries → fresh gating evaluation

- **Batching**: If processing logs, batch episodes (future feature).

## Security

- **PII**: The system automatically redacts the following PII types:
  - Email addresses
  - Phone numbers (when enabled)
  - Social Security Numbers (SSNs)
  - Credit card numbers
  - API keys and secrets (AWS, GitHub, OpenAI, etc.)
  - IP addresses (IPv4 and IPv6)
  - Names and locations (via NER)
  - File paths containing usernames
  
  **Note**: This list may be non-exhaustive. Treat any sensitive data as PII and add additional client-side safeguards where needed. See [PII Redaction docs](../src/utils/pii_redaction.py) for the complete pattern list.

- **API Keys**: Rotate your `X-API-Key` regularly.
- **Namespace**: Always ensure you are using the correct `customer_id` (handled via API key).
