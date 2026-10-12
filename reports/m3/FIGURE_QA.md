# M3 figure QA

- Plot-source validator: 12 PASS, 2 WARN, 0 FAIL. Width warnings are accepted for
  an engineering report bundle (not final journal-column submission). RNG warning
  is a false positive: the RNG only subsamples existing real PLY vertices, never
  synthesizes measurements, errors or point clouds.
- Initial rendered QA was performed with the user-specified Agnes API; outputs
  are recorded as `figure_visual_qa_batch1.json` and `figure_visual_qa_batch2.json`.
  图片内容由 Agnes API 识别。
- Figure 01 passes: fixed physical axis bounds and common reference across dates.
- Figures 02/03 legends are moved outside axes to remove curve overlap. Growth
  and MAE panels start at zero. Figure 03 explicitly describes three training seeds.
- Figure 04 uses symmetric zero-centered bounds, common cm bounds for height and
  width, and independent count units for leaves. Positive/negative interpretation
  is in the title. No significance stars are fabricated when a CI crosses zero.
- Different physical traits legitimately retain separate y units/scales; forcing
  centimetres and leaf counts onto a single numeric range is not scientifically
  meaningful. Panel axis labels explicitly state each trait/unit.
- Figures 05–08 initial visual QA passes. All legends are subsequently external
  to axes for consistent traceability. PNG/TIFF are 600 dpi; SVG has editable text,
  PDF uses embedded TrueType fonts. Large 3D scatters are rasterized only inside
  otherwise vector exports to keep file size manageable.
- Final Agnes rendered recheck of Figures 02/03/04: all PASS; response preserved
  in `figure_visual_qa_final.json`. No unresolved clipping/legend overlap finding
  remains for the deliverable report figures.
