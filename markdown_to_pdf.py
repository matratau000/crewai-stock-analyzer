import markdown
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib.enums import TA_JUSTIFY
import re
import os
import shutil
import glob
import sys
import logging
import base64

def extract_sources(md_content):
    sources = []
    sources_section = re.search(r"## Sources:\s*([\s\S]+)$", md_content)
    if sources_section:
        sources_text = sources_section.group(1).strip()
        source_entries = sources_text.split('\n')
        for entry in source_entries:
            if entry.strip():
                sources.append({"Title": entry.strip(), "Link": "#"})
    return sources

def clean_markdown(content):
    # Split content into lines and strip each line
    lines = [line.strip() for line in content.split('\n')]
    
    # Ensure proper main title
    if lines and lines[0].startswith('Financial Report for Nvidia'):
        lines[0] = '# ' + lines[0]
    
    # Remove duplicate headers and empty lines
    new_lines = []
    for line in lines:
        if new_lines and line == new_lines[-1]:
            continue
        if line or new_lines:  # Keep line if it's not empty or there are previous lines
            new_lines.append(line)
    
    # Join lines back into content
    content = '\n'.join(new_lines)
    
    # Format the Price Analysis section
    price_analysis = re.search(r'## Price Analysis:.*?(?=##|\Z)', content, re.DOTALL)
    if price_analysis:
        price_content = price_analysis.group()
        price_lines = price_content.split('\n')
        formatted_price = "## Price Analysis:\n\n" + "\n".join([f"* {line.strip()}" for line in price_lines if line.strip() and not line.startswith('##')])
        content = content.replace(price_content, formatted_price)
    
    # Ensure Trade Signal is a proper header
    content = re.sub(r'##\s*Trade Signal:', '## Trade Signal:', content)
    
    # Format the Sources section
    sources_section = re.search(r'Sources:(.*?)(?=##|\Z)', content, re.DOTALL)
    if sources_section:
        sources_content = sources_section.group(1)
        sources_lines = sources_content.split('\n')
        formatted_sources = "## Sources:\n\n" + "\n".join([f"* {line.strip()}" for line in sources_lines if line.strip()])
        content = content.replace(sources_section.group(), formatted_sources)
    
    return content.strip()

def markdown_to_pdf(md_file, pdf_file, image_path):
    # Create a temporary copy of the Markdown file
    temp_md_file = md_file + '.temp'
    shutil.copy2(md_file, temp_md_file)
    
    # Read the Markdown content
    with open(temp_md_file, 'r', encoding='utf-8') as file:
        md_content = file.read()
    
    # Clean and process the Markdown content
    md_content = clean_markdown(md_content)
    
    # Extract sources from markdown content
    sources = extract_sources(md_content)
    
    # Remove sources section from markdown content
    md_content = re.sub(r"## Sources:[\s\S]+", "", md_content)
    
    # Write the cleaned content back to the temporary file
    with open(temp_md_file, 'w', encoding='utf-8') as file:
        file.write(md_content)
    
    # Convert Markdown to HTML
    html = markdown.markdown(md_content, extensions=['extra', 'toc'])
    
    # Create PDF
    doc = SimpleDocTemplate(pdf_file, pagesize=letter,
                            rightMargin=72, leftMargin=72,
                            topMargin=72, bottomMargin=18)
    Story = []
    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(name='Justify', alignment=TA_JUSTIFY))
    
    # Process HTML content
    for line in html.split('\n'):
        if line.startswith('<h1>'):
            p = Paragraph(line[4:-5], styles['Title'])
        elif line.startswith('<h2>'):
            p = Paragraph(line[4:-5], styles['Heading2'])
        elif line.startswith('<p>'):
            # Remove img tags from paragraphs
            line = re.sub(r'<img.*?>', '', line)
            if line != '<p></p>':
                p = Paragraph(line[3:-4], styles['Justify'])
            else:
                continue
        elif line.startswith('<ul>'):
            continue
        elif line.startswith('<li>'):
            p = Paragraph('• ' + line[4:-5], styles['Normal'])
        else:
            continue
        Story.append(p)
        Story.append(Spacer(1, 12))
    
    # Add image
    img = Image(image_path, width=6*inch, height=4*inch)
    Story.append(img)
    Story.append(Spacer(1, 12))
    
    # Add sources
    Story.append(Paragraph('Sources:', styles['Heading2']))
    for source in sources:
        p = Paragraph(f"• {source['Title']}", styles['Normal'])
        Story.append(p)
        Story.append(Spacer(1, 6))
    
    doc.build(Story)
    
    # Delete the temporary Markdown file
    os.remove(temp_md_file)

# Usage
if __name__ == "__main__":
    # Find the most recent markdown file
    md_files = glob.glob('*_Week.md')
    if not md_files:
        print("No markdown file found matching the pattern *_Week.md")
        sys.exit(1)
    md_file = max(md_files, key=os.path.getctime)

    # Create the PDF filename based on the markdown filename
    pdf_file = md_file.replace('.md', '.pdf')

    # Find the most recent PNG file
    png_files = glob.glob('*weekly*_chart.png')
    if not png_files:
        print("No PNG file found matching the pattern *weekly*_chart.png")
        sys.exit(1)
    image_path = max(png_files, key=os.path.getctime)

    markdown_to_pdf(md_file, pdf_file, image_path)
    print(f"PDF generated: {pdf_file}")
