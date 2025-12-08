"""
iHR Import Component

Provides UI for bulk importing iHR standard objects.
"""

import streamlit as st
from typing import TYPE_CHECKING, List, Tuple

if TYPE_CHECKING:
    from object_registry import ObjectRegistry

# iHR standard objects for RoboCup@Home
# Format: (name, display_name, category, is_heavy, is_tiny, has_liquid, size_cm, remarks)
IHR_OBJECTS: List[Tuple[str, str, str, bool, bool, bool, str, str]] = [
    ("noodles", "Noodles", "Food", False, False, False, None, ""),
    ("tea_bag", "Tea Bag", "Food", False, False, False, None, ""),
    ("potato_chips", "Potato Chips", "Food", False, False, False, None, ""),
    ("gummy", "Gummy", "Food", False, False, False, None, ""),
    ("redbull", "Redbull", "Drink", True, False, True, None, "Heavy Item, Liquid included"),
    ("aquarius", "Aquarius", "Drink", True, False, True, None, "Heavy Item, Liquid included"),
    ("lychee", "Lychee", "Drink", True, False, True, None, "Heavy Item, Liquid included"),
    ("coffee", "Coffee", "Drink", True, False, True, None, "Heavy Item, Liquid included"),
    ("detergent", "Detergent", "Kitchen Item", False, False, False, None, "Without content"),
    ("cup", "Cup", "Kitchen Item", False, False, False, None, ""),
    ("lunch_box", "Lunch Box", "Kitchen Item", False, False, False, None, ""),
    ("bowl", "Bowl", "Kitchen Item", False, False, False, None, ""),
    ("dice", "Dice", "Task Item", False, True, False, "1.6x1.6x1.6", "Tiny Item"),
    ("light_bulb", "Light Bulb", "Task Item", False, False, False, None, "Without content"),
    ("block", "Block", "Task Item", False, False, False, None, ""),
    ("glue_gun", "Glue Gun", "Task Item", False, False, False, None, "Without plastic container"),
    ("shopping_bag", "Shopping Bag", "Bag", False, False, False, None, ""),
]


def render_ihr_import(registry: "ObjectRegistry"):
    """
    Render iHR standard objects import section.

    Args:
        registry: ObjectRegistry instance
    """
    st.markdown("---")
    st.subheader("Quick Import: iHR Object List")

    # Show preview of objects
    with st.expander("Preview iHR Objects", expanded=False):
        for name, display, category, heavy, tiny, liquid, size, remarks in IHR_OBJECTS:
            props = []
            if heavy:
                props.append("Heavy")
            if tiny:
                props.append("Tiny")
            if liquid:
                props.append("Liquid")
            props_str = f" ({', '.join(props)})" if props else ""
            st.write(f"• **{display}** [{category}]{props_str}")

    if st.button("Import iHR Standard Objects"):
        count = _import_ihr_objects(registry)
        st.success(f"Imported {count} objects from iHR list")
        st.rerun()


def _import_ihr_objects(registry: "ObjectRegistry") -> int:
    """
    Import iHR standard objects into registry.

    Args:
        registry: ObjectRegistry instance

    Returns:
        Number of objects imported
    """
    from object_registry import RegisteredObject, ObjectProperties

    count = 0
    for name, display, cat, heavy, tiny, liquid, size, remarks in IHR_OBJECTS:
        # Skip if already exists
        if registry.get_object_by_name(name):
            continue

        obj = RegisteredObject(
            id=registry.get_next_id(),
            name=name,
            display_name=display,
            category=cat,
            target_samples=100,
            remarks=remarks,
            properties=ObjectProperties(
                is_heavy=heavy,
                is_tiny=tiny,
                has_liquid=liquid,
                size_cm=size,
            ),
        )
        registry.add_object(obj)
        count += 1

    return count
