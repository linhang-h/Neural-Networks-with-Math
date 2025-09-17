<!--
Add here global page variables to use throughout your website.
-->
+++
author = "Linhang Huang"
mintoclevel = 2

# please do read the docs on deployment to avoid common issues: https://franklinjl.org/workflow/deploy/#deploying_your_website
prepath = "Neural-Networks-with-Math"

# RSS (the website_{title, descr, url} must be defined to get RSS)
generate_rss = true
website_title = "Neural Networks with Math"
website_descr = "Let's leanr neural networks with MATH"
website_url   = "https://linhang-h.github.io/Neural-Networks-with-Math/"
+++

<!--
Add here global latex commands to use throughout your pages.
-->
\newcommand{\R}{\mathbb R}
\newcommand{\C}{\mathbb C}
\newcommand{\scal}[1]{\langle #1 \rangle}
\newcommand{\k}{\mathbb k}

\newcommand{\block}[2]{
  @@note
    @@title
        #1
    @@
    @@content
        #2
    @@
  @@
}
