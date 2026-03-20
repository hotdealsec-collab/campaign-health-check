# Campaign Health Check

Lightweight Streamlit app for monitoring campaign health using performance and measurement confidence signals.

## Overview

This app provides a simple monitoring view for marketing campaigns based on:

- Growth Health Score
- Measurement Confidence Score
- Basic campaign-level filtering
- Visual scatter plot for quick positioning

This version is intentionally lightweight and designed for hypothesis checking and operational monitoring.

## Features

- Upload CSV campaign data
- View average growth and measurement indicators
- Filter by channel, campaign, and OS
- Visualize campaign positions in a scatter plot
- Review campaign-level metrics in a table
- Download filtered campaign data as CSV

## Required Columns

The uploaded CSV must include the following columns:

- channel
- campaign
- os
- spend
- installs
- activated_users
- d1_retention
- d3_retention
- d7_retention
- revenue
- skan_only
- strategic_channel
- period_start
- period_end

## Run locally

```bash
pip install -r requirements.txt
streamlit run app.py
