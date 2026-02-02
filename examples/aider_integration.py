# Vritti Integration with Aider
# =================================
# Shows how to integrate Vritti memory with Aider coding agent

import sys
import os
from pathlib import Path

# Add vritti examples to path
sys.path.insert(0, str(Path(__file__).parent))

from coding_agent_integration import VrittiAgent
import asyncio


class AiderWithVritti:
    """
    Wrapper that adds Vritti memory to Aider coding agent.
    
    Intercepts Aider's commands and adds gating + learning.
    """
    
    def __init__(self, api_key: str):
        self.vritti = VrittiAgent(api_key=api_key)
        self.execution_history = []
    
    async def execute_command_safely(
        self,
        command: str,
        goal: str,
        context: dict = None
    ) -> tuple[bool, str]:
        """
        Execute command with Vritti gating.
        
        Returns:
            (success: bool, output: str)
        """
        # Step 1: Check with Vritti before executing
        check = await self.vritti.check_action(
            action=command,
            goal=goal,
            tool="shell",
            context=context or {}
        )
        
        if not check['safe']:
            print(f"🛑 Vritti BLOCKED: {check['reason']}")
            if check['alternative']:
                print(f"💡 Suggested: {check['alternative']}")
            return False, f"Blocked by Vritti: {check['reason']}"
        
        # Step 2: Execute (Aider would do this)
        print(f"✅ Vritti PROCEED - executing: {command}")
        
        # Simulate execution (in real integration, Aider executes command)
        import subprocess
        try:
            result = subprocess.run(
                command,
                shell=True,
                capture_output=True,
                text=True,
                timeout=30
            )
            
            success = result.returncode == 0
            output = result.stdout if success else result.stderr
            
            # Step 3: Track in history
            self.execution_history.append({
                'command': command,
                'goal': goal,
                'success': success,
                'output': output
            })
            
            # Step 4: If failed, learn from it
            if not success:
                await self.vritti.learn_from_failure(
                    goal=goal,
                    actions=[command],
                    error=output,
                    tool_chain=["shell"],
                    context=context
                )
                print(f"📝 Learned from failure - won't repeat this mistake")
            
            return success, output
            
        except Exception as e:
            # Learn from exception too
            await self.vritti.learn_from_failure(
                goal=goal,
                actions=[command],
                error=str(e),
                tool_chain=["shell"],
                context=context
            )
            return False, str(e)


async def demo_aider_integration():
    """
    Demo showing Aider with Vritti memory.
    """
    print("=" * 70)
    print("Aider + Vritti Integration Demo")
    print("=" * 70)
    
    aider = AiderWithVritti(api_key="em_live_test_key_12345")
    
    # Scenario 1: Try to delete important files (should be blocked)
    print("\n📋 Task: Clean up project files")
    success, output = await aider.execute_command_safely(
        command="rm -rf .git",
        goal="Clean up version control files",
        context={"project": "myproject", "has_commits": True}
    )
    
    # Scenario 2: Safe command (should be allowed)
    print("\n📋 Task: List directory contents")
    success, output = await aider.execute_command_safely(
        command="ls -la",
        goal="Check project structure",
        context={"project": "myproject"}
    )
    
    print(f"\nResult: {'✅ Success' if success else '❌ Failed'}")
    
    # Scenario 3: Command that might fail
    print("\n📋 Task: Install non-existent package")
    success, output = await aider.execute_command_safely(
        command="pip install nonexistent-package-xyz123",
        goal="Install dependencies",
        context={"python_version": "3.9"}
    )
    
    # Scenario 4: Same command again (should be blocked by Vritti)
    print("\n📋 Task: Try installing same package again")
    print("(Vritti should remember the previous failure)")
    success, output = await aider.execute_command_safely(
        command="pip install nonexistent-package-xyz123",
        goal="Install dependencies",
        context={"python_version": "3.9"}
    )


if __name__ == "__main__":
    asyncio.run(demo_aider_integration())
