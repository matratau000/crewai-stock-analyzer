import pdfkit
import sys
import markdown

def markdown_to_pdf(md_file):
    # Define CSS styles for better formatting
    css_styles = """
    <style>
        body {
            font-family: Arial, sans-serif;
            margin: 20px;
        }
        h1, h2, h3, h4, h5, h6 {
            color: #2E86C1;
        }
        p {
            margin: 10px 0;
        }
        table {
            width: 100%;
            border-collapse: collapse;
            margin: 20px 0;
        }
        table, th, td {
            border: 1px solid #ddd;
        }
        th, td {
            padding: 8px;
            text-align: left;
        }
        th {
            background-color: #f2f2f2;
        }
        a {
            color: #1A5276;
            text-decoration: none;
        }
        a:hover {
            text-decoration: underline;
        }
        blockquote {
            margin: 10px 20px;
            padding: 10px;
            background-color: #f9f9f9;
            border-left: 10px solid #ccc;
        }
        code {
            font-family: Consolas, "Courier New", monospace;
            background-color: #f4f4f4;
            padding: 2px 4px;
            border-radius: 4px;
        }
        pre {
            background-color: #f4f4f4;
            padding: 10px;
            border-radius: 4px;
            overflow-x: auto;
        }
        img {
            max-width: 100%;
            height: auto;
        }
    </style>
    """
    # Convert Markdown to HTML
    with open(md_file, 'r') as file:
        md_content = file.read()
    
    # Remove "tm" ASCII characters
    md_content = md_content.replace("â„¢", "")

    html_content = markdown.markdown(md_content, extensions=['extra', 'smarty'])
    
    # Ensure links are properly formatted with titles included
    for a in soup.find_all('a'):
        if not a.get('title'):
            a['title'] = a.get('href')
        # Add title to the link text if not present
        if a.string and a['title'] not in a.string:
            a.string = f"{a.string} ({a['title']})"
    from bs4 import BeautifulSoup

    soup = BeautifulSoup(html_content, 'html.parser')
    for a in soup.find_all('a'):
        if not a.get('title'):
            a['title'] = a.get('href')
    html_content = str(soup)

    # Combine CSS styles with HTML content
    html_content = f"{css_styles}{html_content}"

    # Define the output PDF file name
    pdf_file = md_file.replace('.md', '.pdf')
    
    # Convert HTML to PDF
    pdfkit.from_string(html_content, pdf_file)
    print(f"PDF generated: {pdf_file}")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python3 markdown_to_pdf.py <markdown_file>")
        sys.exit(1)
    
    markdown_file = sys.argv[1]
    markdown_to_pdf(markdown_file)
