#!/usr/bin/env python3
"""Quick test to verify vulkan_backend imports correctly."""

try:
    import rasptorch.vulkan_backend as vk
    print("✓ rasptorch.vulkan_backend imported successfully")
    print(f"✓ _HAS_VULKAN = {vk._HAS_VULKAN}")
    if vk._VULKAN_DISABLED_REASON:
        print(f"  Reason: {vk._VULKAN_DISABLED_REASON}")
    print("✓ All syntax errors fixed!")
except Exception as e:
    print(f"✗ Import failed: {e}")
    import traceback
    traceback.print_exc()
