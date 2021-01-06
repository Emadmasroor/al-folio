---
layout: page
title: news + events
permalink: /news/
description: Here's what I have been up to recently.
nav: true
---

{% assign sorted_news = site.news | sort:"date" | reverse %}
{% for news in sorted_news %}
<article class="post-content">
 {{ news.date | date: "%b %d, %Y"}}
 {{ news.content }}
<article class="post-content">
{% endfor %}

