#!/usr/bin/env python3
"""Script to append Vietnamese translations to design document"""

# Read the Vietnamese translations file
with open('.kiro/specs/vietnamese-prompt-localization/vietnamese-translations.md', 'r', encoding='utf-8') as f:
    translations_content = f.read()

# Extract content starting from section 2 (Entity Extraction User Prompt)
start_marker = "### 2. Entity Extraction User Prompt (Vietnamese)"
start_index = translations_content.find(start_marker)

if start_index != -1:
    remaining_content = translations_content[start_index:]
    
    # Append to design document
    with open('.kiro/specs/vietnamese-prompt-localization/design.md', 'a', encoding='utf-8') as f:
        f.write("\n\n")
        f.write(remaining_content)
    
    print("Successfully appended Vietnamese translations to design document")
else:
    print("Could not find start marker in translations file")
