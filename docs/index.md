---
hide:
  - toc
---

<div class="landing-page-marker" aria-hidden="true"></div>

<section class="rem-hero" aria-labelledby="rem-hero-title">
  <div class="rem-hero__copy">
    <span class="rem-hero__eyebrow">REM'S LLM NOTES</span>
    <h1 id="rem-hero-title">LLM 学习笔记</h1>
    <p class="rem-hero__intro">欢迎来到雷姆哈滋邦德的频道!!! 这里收录了我平时学习LLM、RL与Post-Train的学习笔记。欢迎大家学习、指正!! (正文保留 Markdown 原稿，数学公式使用 LaTeX 编写)</p>
  </div>
  <a class="rem-hero__visual" href="https://re-zero-anime.jp/tv/character/" target="_blank" rel="noopener" aria-label="前往 Re:Zero 动画官网角色页面">
    <span class="rem-hero__halo" aria-hidden="true"></span>
    <img src="assets/images/theme/rem-hero.webp" alt="雷姆（Re:Zero）">
    <span class="rem-hero__name" aria-hidden="true">REM</span>
  </a>
</section>

## 快速入口

<nav class="portal-grid" aria-label="站点主要入口">
  <a class="portal-card portal-card--current" href="#rem-hero-title">
    <span class="portal-card__index">01</span>
    <strong>简介</strong>
    <span>了解这个站点记录什么，以及笔记的整理方式。</span>
  </a>
  <a class="portal-card" href="categories/">
    <span class="portal-card__index">02</span>
    <strong>分类</strong>
    <span>先选择学习方向，再进入该分类的文章目录。</span>
  </a>
  <a class="portal-card" href="updates/">
    <span class="portal-card__index">03</span>
    <strong>最新更新</strong>
    <span>按发布日期查看更新，并直接进入具体文章。</span>
  </a>
  <a class="portal-card" href="download/">
    <span class="portal-card__index">04</span>
    <strong>下载入口</strong>
    <span>下载完整仓库、单篇 Markdown 原稿与相关图片。</span>
  </a>
</nav>

## 最新更新

<div class="updates-list">
  <article class="latest-note-card">
    <div class="latest-note-card__meta">
      <time datetime="2026-07-15">2026.07.15</time>
      <span>后训练基础</span>
    </div>
    <h3><a href="post-training-basics/pytorch-training-causallm/">PyTorch 训练底层与 CausalLM 手撕</a></h3>
    <p>从 Tensor、计算图和 AdamW 开始，串起 CausalLM 的前向传播、Loss、反向传播与参数更新。</p>
    <a class="latest-note-card__link" href="post-training-basics/pytorch-training-causallm/">直接阅读 →</a>
  </article>

  <article class="latest-note-card">
    <div class="latest-note-card__meta">
      <time datetime="2026-07-14">2026.07.14</time>
      <span>从数学讲清后训练</span>
    </div>
    <h3><a href="post-training/dpo-implicit-kl/">DPO 为什么只做偏好分类，却“自带” KL 约束？</a></h3>
    <p>从 KL-Regularized RL 出发，推导最优策略、Reward 表达与 DPO Loss 之间的关系。</p>
    <a class="latest-note-card__link" href="post-training/dpo-implicit-kl/">直接阅读 →</a>
  </article>

  <article class="latest-note-card">
    <div class="latest-note-card__meta">
      <time datetime="2026-07-14">2026.07.14</time>
      <span>从数学讲清后训练</span>
    </div>
    <h3><a href="post-training/distributional-view/">在分布视角下理解语言模型后训练</a></h3>
    <p>从序列分布与自回归概率树出发，在同一视角下理解 SFT、RL 与 OPD。</p>
    <a class="latest-note-card__link" href="post-training/distributional-view/">直接阅读 →</a>
  </article>
</div>

<p class="section-more"><a href="updates/">查看全部更新记录 →</a></p>

## 下载入口

Markdown 原稿、数学公式和文章图片均保留在 GitHub 仓库中。

[前往下载页面](download.md){ .md-button .md-button--primary }
