EXPRESSION_COLORS = (
    "#EF4444",
    "#22C55E",
    "#A855F7",
    "#FACC15",
    "#3B82F6",
    "#F97316",
    "#94A3B8",
)
EXPRESSION_RGB = tuple(
    tuple(int(color[index : index + 2], 16) for index in (1, 3, 5)) for color in EXPRESSION_COLORS
)
UNCLASSIFIED_RGB = (148, 163, 184)
