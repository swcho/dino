Create a clean flat-design educational infographic, 16:9 landscape, titled "DINO: Three Core Asymmetries" at the top center in a bold sans-serif heading, with a small Korean subtitle underneath reading "DINO 목적함수의 세 가지 비대칭".

Overall layout: a single header band, then three equal vertical panels side by side, read left to right, numbered 1, 2, 3 in large circular badges. A thin horizontal footer strip runs across the bottom. Use a light off-white background, one warm orange accent for the TEACHER side and one cool blue accent for the STUDENT side, plus neutral gray for supporting elements. Generous white space, thin 2px outlines, rounded rectangles, no gradients, no photographs.

PANEL 1, badge "1", panel heading "Teacher sees global only":
Two labeled columns. Left column labeled "TEACHER" in orange, showing two large square image crops labeled "Global 224px". Right column labeled "STUDENT" in blue, showing two large squares labeled "Global 224px" plus a row of eight small squares labeled "Local 96px x8". A curved arrow points from the small local crops up to the teacher's large crop, labeled "local to global". Small monospace code chip at the panel bottom: "teacher(images[:2])".

PANEL 2, badge "2", panel heading "Same view pairs excluded":
A grid matrix, 2 rows by 10 columns. Rows labeled "Teacher u" in orange, columns labeled "Student v" in blue. Fill all cells with light blue checkmarks except the two diagonal cells (row 1 col 1, row 2 col 2), which are gray with a red X and a small label "v = u skipped". Beneath the grid a bold count: "18 loss terms". Small monospace code chip at the panel bottom: "if v == iq: continue".

PANEL 3, badge "3", panel heading "No gradient to teacher":
A vertical loop diagram. Blue "STUDENT" box at bottom, orange "TEACHER" box at top. A blue upward arrow from student to teacher labeled "EMA update". A gray downward dashed arrow from teacher to student labeled "target". A red backward gradient arrow drawn from the loss into the teacher is crossed out with a bold red X and labeled "stop-gradient". A formula chip in the middle reads exactly "theta_t = m*theta_t + (1-m)*theta_s" and beneath it "m: 0.996 to 1.0". Small monospace code chip at the panel bottom: ".detach() / requires_grad=False".

FOOTER strip, single line of three short paired labels separated by vertical dividers, each with a tiny warning triangle icon: "Remove 1: no part-whole learning" | "Remove 2: student copies teacher" | "Remove 3: collapse to constant".

Typography: clean geometric sans-serif, large readable panel headings, short labels only. All text rendered crisply and spelled exactly as given.
