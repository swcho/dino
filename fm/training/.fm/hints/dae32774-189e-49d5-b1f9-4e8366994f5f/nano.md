Create a clean flat-design educational infographic, 16:9 landscape, titled "MultiCropWrapper: 10 Crops, 2 Forwards".

Layout: two stacked horizontal panels separated by a thin divider line. Eye flow is left to right within each panel, and top panel contrasts with bottom panel. A soft light-gray background, white panel cards with rounded corners, one accent red for the top panel and one accent green for the bottom panel, plus neutral blue-gray boxes for the network.

TOP PANEL, labeled on the left edge with the small red header "Naive: one by one":
On the left, a vertical stack of 10 small image thumbnails drawn as simple rounded squares. The top 2 squares are noticeably LARGER and tinted blue, labeled "2 global 224px". The bottom 8 squares are SMALLER and tinted orange, labeled "8 local 96px". From every one of the 10 squares draw a separate thin red arrow pointing right into a single tall gray rounded box labeled "Backbone (ViT)". So exactly 10 red arrows converge on the box. Above the arrow bundle put a bold red badge reading "10 forwards". To the right of the backbone box add a small red warning chip "Slow, GPU idle".

BOTTOM PANEL, labeled on the left edge with the small green header "MultiCropWrapper: grouped":
On the left, the same 10 thumbnails but visually bundled into two brackets. The top bracket wraps the 2 large blue squares and is labeled "(2B, 3, 224, 224)". The bottom bracket wraps the 8 small orange squares and is labeled "(8B, 3, 96, 96)". From each bracket draw ONE thick green arrow into the same gray "Backbone (ViT)" box, so only 2 green arrows total. Above them a bold green badge reading "2 forwards". To the right of the backbone box, show two output slabs, a blue one labeled "(2B, D)" and an orange one labeled "(8B, D)", merging with a small plus-shaped concat node labeled "concat" into one taller two-tone slab labeled "(10B, D)". One arrow from that slab into a single purple rounded box labeled "Head (once)", and one final arrow out to a slab labeled "(10B, K)".

Bottom strip across full width: a slim horizontal note bar with three short chips, evenly spaced: "Same size = same batch", "unique_consecutive + cumsum", "Row order preserved".

Style: modern educational infographic, clean flat vector, no gradients, no photos, generous whitespace, crisp sans-serif labels, consistent icon sizes, high contrast readable text. All labels short and exactly as written above. Aspect ratio 16:9.
