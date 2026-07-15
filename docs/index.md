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

## 内容分类

<div class="grid cards" markdown>

-   **从数学讲清后训练**

    从目标函数、概率分布与优化过程出发，理解后训练算法为什么成立。

    [进入分类](categories/math-post-training.md)

-   **后训练基础**

    补齐模型结构、训练链路与实现基础，为后续学习 SFT、DPO、PPO、GRPO 建立统一底座。

    [进入分类](categories/post-training-basics.md)

</div>

## 最新收录

<article class="latest-note-card">
  <div class="latest-note-card__meta">
    <time datetime="2026-07-15">2026.07.15</time>
    <span>后训练基础</span>
  </div>
  <h3><a href="post-training-basics/pytorch-training-causallm/">PyTorch 训练底层与 CausalLM 手撕</a></h3>
  <p>从 Tensor、Parameter、计算图和 AdamW 开始，串起 CausalLM 的前向传播、Loss、反向传播与参数更新，并进一步手写最小 Llama 风格 Transformer。</p>
  <a class="latest-note-card__link" href="post-training-basics/pytorch-training-causallm/">阅读文章 →</a>
</article>

## 获取原文

- [下载全部笔记（ZIP）](https://github.com/kaining-never-stop/llm-learning-notes/archive/refs/heads/main.zip)
- [查看 GitHub 仓库](https://github.com/kaining-never-stop/llm-learning-notes)
- [查看全部 Markdown 原文](https://github.com/kaining-never-stop/llm-learning-notes/tree/main/docs)

更多下载方式见[获取笔记](download.md)。
