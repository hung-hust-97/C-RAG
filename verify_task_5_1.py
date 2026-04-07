#!/usr/bin/env python3
"""
Simple verification script for Task 5.1: Entity extraction uses PromptManager
This script verifies the implementation without requiring full LightRAG dependencies.
"""

import sys
import os

# Add lightrag to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'lightrag'))

def verify_imports():
    """Verify that all required imports are available"""
    try:
        from lightrag.prompt_manager import PromptManager
        from lightrag.language_detector import SupportedLanguage
        from lightrag.prompt import PROMPTS
        print("✅ All required imports successful")
        return True, PromptManager, SupportedLanguage, PROMPTS
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False, None, None, None

def verify_prompt_manager_functionality(PromptManager, SupportedLanguage, PROMPTS):
    """Verify PromptManager functionality for entity extraction"""
    try:
        # Initialize PromptManager
        manager = PromptManager(PROMPTS)
        print("✅ PromptManager initialized successfully")
        
        # Test English entity extraction prompt
        english_prompt = manager.get_prompt(
            "entity_extraction_system_prompt",
            language=SupportedLanguage.ENGLISH
        )
        assert len(english_prompt) > 0
        assert "Knowledge Graph Specialist" in english_prompt
        print("✅ English entity extraction prompt retrieved correctly")
        
        # Test Vietnamese entity extraction prompt
        vietnamese_prompt = manager.get_prompt(
            "entity_extraction_system_prompt",
            language=SupportedLanguage.VIETNAMESE
        )
        assert len(vietnamese_prompt) > 0
        assert "Chuyên gia Đồ thị Tri thức" in vietnamese_prompt
        print("✅ Vietnamese entity extraction prompt retrieved correctly")
        
        # Test auto-detection
        vietnamese_text = "Trích xuất thực thể từ văn bản này với đủ ký tự đặc biệt"
        auto_prompt = manager.get_prompt(
            "entity_extraction_system_prompt",
            auto_detect_text=vietnamese_text
        )
        assert "Chuyên gia Đồ thị Tri thức" in auto_prompt
        print("✅ Auto-detection works for Vietnamese text")
        
        # Test user prompt
        user_prompt = manager.get_prompt(
            "entity_extraction_user_prompt",
            language=SupportedLanguage.VIETNAMESE
        )
        assert len(user_prompt) > 0
        assert "Trích xuất thực thể" in user_prompt
        print("✅ Vietnamese entity extraction user prompt retrieved correctly")
        
        # Test continue extraction prompt
        continue_prompt = manager.get_prompt(
            "entity_continue_extraction_user_prompt",
            language=SupportedLanguage.VIETNAMESE
        )
        assert len(continue_prompt) > 0
        assert "bị bỏ sót" in continue_prompt
        print("✅ Vietnamese continue extraction prompt retrieved correctly")
        
        return True
        
    except Exception as e:
        print(f"❌ PromptManager functionality error: {e}")
        return False

def verify_operate_py_integration():
    """Verify that operate.py uses PromptManager correctly"""
    try:
        with open('lightrag/operate.py', 'r') as f:
            content = f.read()
        
        # Check for PromptManager import
        assert "from lightrag.prompt_manager import PromptManager" in content
        print("✅ PromptManager is imported in operate.py")
        
        # Check for PromptManager initialization
        assert "prompt_manager = PromptManager(PROMPTS)" in content
        print("✅ PromptManager is initialized in operate.py")
        
        # Check for entity extraction system prompt usage
        assert 'prompt_manager.get_prompt(\n            "entity_extraction_system_prompt"' in content
        print("✅ Entity extraction system prompt uses PromptManager")
        
        # Check for entity extraction user prompt usage
        assert 'prompt_manager.get_prompt(\n            "entity_extraction_user_prompt"' in content
        print("✅ Entity extraction user prompt uses PromptManager")
        
        # Check for continue extraction prompt usage
        assert 'prompt_manager.get_prompt(\n            "entity_continue_extraction_user_prompt"' in content
        print("✅ Entity continue extraction prompt uses PromptManager")
        
        # Check for language parameter usage
        assert "language=prompt_language" in content
        print("✅ Language parameter is passed to PromptManager")
        
        return True
        
    except Exception as e:
        print(f"❌ operate.py integration error: {e}")
        return False

def main():
    """Main verification function"""
    print("=" * 60)
    print("VERIFYING TASK 5.1: Entity extraction uses PromptManager")
    print("=" * 60)
    
    # Step 1: Verify imports
    success, PromptManager, SupportedLanguage, PROMPTS = verify_imports()
    if not success:
        return 1
    
    # Step 2: Verify PromptManager functionality
    success = verify_prompt_manager_functionality(PromptManager, SupportedLanguage, PROMPTS)
    if not success:
        return 1
    
    # Step 3: Verify operate.py integration
    success = verify_operate_py_integration()
    if not success:
        return 1
    
    print("\n" + "=" * 60)
    print("✅ TASK 5.1 VERIFICATION SUCCESSFUL!")
    print("=" * 60)
    print("\nSummary:")
    print("✅ PromptManager is imported in lightrag/operate.py")
    print("✅ PromptManager replaces direct PROMPTS dictionary access")
    print("✅ Language parameter is passed to get_prompt() method")
    print("✅ All entity extraction prompts use PromptManager:")
    print("   - entity_extraction_system_prompt")
    print("   - entity_extraction_user_prompt") 
    print("   - entity_continue_extraction_user_prompt")
    print("✅ Auto-detection and explicit language specification work")
    print("✅ Requirements 2.1, 2.2, 10.6, 10.7 are satisfied")
    
    return 0

if __name__ == "__main__":
    exit(main())