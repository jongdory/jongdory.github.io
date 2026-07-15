---
layout: homepage
---

## About Me

I am a Ph.D. student at ...


## News

<ul id="news-list">
  <li><strong>[Jun. 2026]</strong> A paper (PS-MIT) is accepted to ECCV 2026.</li>
  <li><strong>[Jun. 2026]</strong> A paper (AgeCMBNet) is accepted to MICCAI 2026.</li>
  <li><strong>[Mar. 2026]</strong> A paper (<a href="https://arxiv.org/pdf/2604.02748">LLaBIT</a>) is accepted to ICPR 2026.</li>
  <li><strong>[Mar. 2026]</strong> A paper (<a href="https://www.sciencedirect.com/science/article/pii/S016926072600012X">VPT-Med</a>) is accepted to CMPB.</li>
  <li><strong>[Sep. 2025]</strong> A paper (<a href="https://www.sciencedirect.com/science/article/pii/S0895611125001600">Vessel-Diffusion</a>) is accepted to CMIG.</li>
  <li><strong>[Jun. 2025]</strong> 3 papers (<a href="https://papers.miccai.org/miccai-2025/paper/0621_paper.pdf">Latent-HE</a> <a href="https://papers.miccai.org/miccai-2025/paper/0187_paper.pdf">BP-nnUNet</a> <a href="https://papers.miccai.org/miccai-2025/paper/2349_paper.pdf">PMIL</a>) are accepted to MICCAI 2025.</li>
  <li class="news-extra" style="display:none;"><strong>[Oct. 2025]</strong> A paper (<a href="https://www.sciencedirect.com/science/article/pii/S0169260725002986">CASCA</a>) is accepted to CMPB.</li>
  <li class="news-extra" style="display:none;"><strong>[May. 2025]</strong> A paper (<a href="https://openaccess.thecvf.com/content/WACV2025/papers/Kim_Tumor_Synthesis_Conditioned_on_Radiomics_WACV_2025_paper.pdf">TS-Radiomics</a>) is accepted to WACV 2025.</li>
  <li class="news-extra" style="display:none;"><strong>[Sep. 2024]</strong> A paper (<a href="https://openaccess.thecvf.com/content/ACCV2024/papers/Kim_Domain_Aware_Multi-Task_Pre-Training_of_3D_Swin_Transformer_for_Brain_ACCV_2024_paper.pdf">DAMT</a>) is accepted to ACCV 2024.</li>
  <li class="news-extra" style="display:none;"><strong>[Oct. 2023]</strong> A paper (<a href="https://openaccess.thecvf.com/content/WACV2024/papers/Kim_Adaptive_Latent_Diffusion_Model_for_3D_Medical_Image_to_Image_WACV_2024_paper.pdf">ALDM</a>) is accepted to WACV 2024.</li>
</ul>
<button id="news-toggle-btn" style="margin-bottom:10px;">Show More</button>
<script>
  document.addEventListener("DOMContentLoaded", function () {
    const extras = document.querySelectorAll('.news-extra');
    const btn = document.getElementById('news-toggle-btn');
    let expanded = false;
    function updateView() {
      extras.forEach(e => { e.style.display = expanded ? '' : 'none'; });
      btn.textContent = expanded ? 'Show Less' : 'Show More';
    }
    btn.addEventListener('click', function () {
      expanded = !expanded;
      updateView();
    });
    updateView();
  });
</script>

{% include_relative _includes/publications.md %}
