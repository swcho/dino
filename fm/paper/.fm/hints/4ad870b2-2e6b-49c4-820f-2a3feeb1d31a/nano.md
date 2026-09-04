Create a clean flat-design educational infographic explaining the DINO multi-crop strategy. Aspect ratio 4:3, landscape. Title at the top center in bold sans-serif: "Multi-Crop: Local-to-Global" with a smaller Korean subtitle underneath: "멀티크롭 전략".

Layout: three panels flowing left to right, connected by arrows, with a clear left-to-right reading direction.

LEFT PANEL — "Source Image / 원본 이미지". Show one large photo-style rectangle of a simple illustrated dog on grass, drawn in flat vector style. On top of it, draw crop rectangles as colored outlines: two LARGE blue rectangles, each covering more than half of the image area, labeled "Global 1" and "Global 2"; and six SMALL orange rectangles scattered over different small parts of the image (ear, eye, paw, grass, tail, nose), each covering a tiny area. A small caption under the panel: "RandomResizedCrop".

MIDDLE PANEL — two stacked groups of cropped thumbnails.
Top group, blue theme, boxed and labeled "2 Global Views" with sub-label "224x224 · scale 0.4-1.0 · large area (>50%)": exactly two big blue-bordered square thumbnails showing wide views of the whole dog.
Bottom group, orange theme, boxed and labeled "Many Local Views" with sub-label "96x96 · scale 0.05-0.4 · small area (<50%)": eight visibly smaller orange-bordered square thumbnails showing tiny zoomed-in patches. Make the size contrast between the blue and orange thumbnails dramatic and obvious.

RIGHT PANEL — two network boxes stacked vertically.
Top box, gray-blue, labeled "Student g_s". Bottom box, gray-green, labeled "Teacher g_t (EMA)".
Arrows: draw BOTH a blue arrow (from the global group) and an orange arrow (from the local group) into the Student box, bundled and labeled "All crops -> Student". Draw ONLY a blue arrow from the global group into the Teacher box, labeled "Only global -> Teacher". Add a small red crossed-out arrow from the local group toward the Teacher box with a circle-slash icon, labeled "No local crops".
Between the two network boxes place a cross-entropy loss node labeled "Cross-entropy loss", with a dashed EMA arrow curving from Student up to Teacher labeled "EMA update".

BOTTOM BAND — a full-width highlighted strip with a large centered arrow going from a small orange square to a big blue square, and the caption in bold: "Predict the whole from a part" with Korean gloss "부분에서 전체를 예측" and a small tag: "local-to-global correspondence".

Style: clean flat design, educational infographic, minimal vector illustration, generous white space, off-white background, consistent 2-color coding throughout (blue = global, orange = local), thin rounded outlines, modern geometric sans-serif labels, no photorealism, no clutter, no gradients, all text crisp and correctly spelled.
