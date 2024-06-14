import markdown
import pdfkit
import base64
import os
import re

# Function to extract sources from the markdown content
def extract_sources(md_content):
    # Assuming sources are provided in the format:
    # "## Sources:\n [{'Title': '...', 'Link': '...'}, ...]"
    sources = []
    sources_section = re.search(r"## Sources:\n(.+)", md_content, re.DOTALL)
    if sources_section:
        sources_text = sources_section.group(1).strip()
        sources = eval(sources_text)  # Converts string representation of list to list
    return sources

# Paths to your files
markdown_file = './Financial_Report_Apple_Week.md'
image_file = './Apple_weekly_chart.png'
output_pdf = './Financial_Report_Apple_Week.pdf'

# Read the markdown file
with open(markdown_file, 'r') as file:
    md_content = file.read()

# Extract sources from markdown content
sources = extract_sources(md_content)

# Remove sources section from markdown content
md_content = re.sub(r"## Sources:\n(.+)", "", md_content, flags=re.DOTALL)

# Convert markdown to HTML
html_content = markdown.markdown(md_content)

# Convert image to base64
with open(image_file, 'rb') as img_file:
    encoded_image = base64.b64encode(img_file.read()).decode('utf-8')

# Generate HTML for sources
sources_html = "<h2>Sources:</h2><ul>"
for source in sources:
    title = source.get("Title", "No title provided")
    link = source.get("Link", "#")
    sources_html += f'<li><a href="{link}">{title}</a></li>'
sources_html += "</ul>"

# HTML template with the image embedded and dynamic sources
html_with_image = f"""
<!DOCTYPE html>
<html>
<head>
    <style>
        body {{
            font-family: Arial, sans-serif;
            line-height: 1.6;
            margin: 20px;
        }}
        h1, h2, h3 {{
            color: #333;
        }}
        img {{
            display: block;
            margin: 20px auto;
            max-width: 100%;
            height: auto;
        }}
        .content {{
            max-width: 800px;
            margin: auto;
        }}
        .sources {{
            font-size: 0.8em;
            margin-top: 20px;
        }}
    </style>
</head>
<body>
    <div class="content">
        {html_content}
        <img src="data:image/png;base64,{encoded_image}" alt="Apple Weekly Chart" />
        <div class="sources">
            {sources_html}
        </div>
    </div>
</body>
</html>
"""

# Convert the HTML to PDF
pdfkit.from_string(html_with_image, output_pdf)
