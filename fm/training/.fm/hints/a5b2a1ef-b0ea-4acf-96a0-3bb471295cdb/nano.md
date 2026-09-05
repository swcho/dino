Create a clean flat-design educational infographic, 16:9 landscape, titled "DINO Multi-Crop Augmentation" at the top center in bold dark sans-serif type, with the small subtitle "One image, three pipelines" beneath it.

Layout: a single source photo thumbnail sits at the top center, just below the title, labeled "Original Image". Three thin connector arrows fan out downward from it into three vertical pipeline columns of equal width that fill the rest of the canvas. The eye flows top-to-bottom through each column.

Column headers, as colored rounded pill badges:
- Left column, blue pill: "Global 1 - 224px"
- Middle column, orange pill: "Global 2 - 224px"
- Right column, gray pill: "Local x8 - 96px"

Each column is a vertical stack of rounded rectangle step boxes connected by short downward arrows. Steps are horizontally aligned across the three columns so they read as rows for easy comparison.

Shared steps (rows 1 to 4) are drawn in muted light gray boxes with dark gray text, identical in all three columns:
- Row 1: "RandomResizedCrop" with the small scale label underneath - "scale 0.4-1.0" in the left column, "scale 0.4-1.0" in the middle column, "scale 0.05-0.4" in the right column
- Row 2: "Horizontal Flip p=0.5"
- Row 3: "ColorJitter p=0.8"
- Row 4: "Grayscale p=0.2"

Differing steps are the visual focus, drawn as filled, saturated, slightly larger boxes with white bold text and a soft drop shadow:
- Row 5 GaussianBlur: left column bright blue box "GaussianBlur p=1.0"; middle column pale orange outlined box "GaussianBlur p=0.1"; right column medium gray box "GaussianBlur p=0.5"
- Row 6 Solarization: middle column only, a bright orange box "Solarization p=0.2"; the left and right columns show a faint dashed empty placeholder box with the small gray label "none"

Final shared row: a muted gray box "Normalize" in all three columns.

At the bottom of each column place a small square example thumbnail illustrating the visual result: left = a visibly blurred soft crop; middle = a sharp crop with inverted bright tones (solarized, bright areas turned dark); right = a small tight close-up crop. Label them in small text: "blurred view", "solarized view", "small patches".

Bottom band, a full-width light strip with a lightbulb icon and the caption "Asymmetric views block low-level shortcuts" in dark text, plus a smaller gray line "from BYOL - 비대칭 증강".

Style: clean flat vector design, generous white space, off-white background, thin 1px connector lines with small arrowheads, rounded corners, restrained palette of blue (#3B82F6), orange (#F97316), and neutral grays. Crisp legible sans-serif labels, no clutter, no long sentences, no watermark. Aspect ratio 16:9.
