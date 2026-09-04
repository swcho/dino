Create a clean flat-design educational infographic, 16:9 landscape, titled "Multi-Head Attention: Fan-Out to Fan-In" with a small subtitle "concat heads, then project by W^O". Layout is a single left-to-right horizontal flow of four labeled panels connected by thick arrows, so the eye reads the fan-out and fan-in shape as one pipeline. Light off-white background, one blue accent for the shared token matrix, six distinct pastel colors (violet, teal, amber, coral, green, indigo) that each stay assigned to the same head across every panel, dark grey text, thin rounded outlines, generous whitespace, no photographic texture, no 3D.

Panel 1 (far left, narrow): one tall blue rectangle made of stacked horizontal rows, labeled below "Tokens Z" and "N x D". A small caption under it: "one shared input".

Fan-out arrows: from Panel 1, six diverging arrows spread outward toward Panel 2, drawn as a visible fan so the split is obvious. Label the arrow bundle "split into heads" with a small formula chip "d_h = D / heads".

Panel 2 (six small stacked cards, one per head, each in its own head color): each card contains three tiny narrow rectangles side by side labeled "Q" "K" "V" and, beside them, a small square grid icon labeled "A_h". Label the top card "Head 1" and the bottom card "Head 6", with a vertical ellipsis between. One caption for the whole panel: "each head, its own view" and a small chip "O_h = A_h V_h". Each card also carries a thin width tag "N x d_h".

Fan-in arrows: six converging arrows from the head cards into Panel 3.

Panel 3: one wide horizontal bar divided into six equal side-by-side segments, each segment filled with its matching head color and separated by thin white gaps, labeled above "Concat [O_1 || ... || O_6]" and below "N x D  (width preserved / 폭 보존)". Add a small bracket under the bar spanning one segment marked "d_h = 64" and a second bracket spanning the whole bar marked "D = 384", so the reader sees 6 x d_h = D.

Panel 4 (far right): a rounded box labeled "W^O  Linear D to D" with an icon of crossing diagonal lines inside suggesting mixing, then a final arrow to one solid violet-to-blue gradient bar with no visible segment seams, labeled "MHSA(Z)" and "N x D  (heads mixed / 혼합)". Small caption: "not concat alone".

Along the bottom, a slim full-width footer strip with three short takeaways separated by dots: "Same width in and out", "Heads stay separate until concat", "W^O blends the heads".
