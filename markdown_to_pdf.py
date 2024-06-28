import pdfkit
import sys
import markdown

def markdown_to_pdf(md_file):
    # Convert Markdown to HTML
    with open(md_file, 'r') as file:
        md_content = file.read()
    
    html_content = markdown.markdown(md_content)
    
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

def markdown_to_pdf(md_file):
    # Convert Markdown to HTML
    with open(md_file, 'r') as file:
        md_content = file.read()
    
    html_content = markdown.markdown(md_content)
    
    # Define the output PDF file name
    pdf_file = md_file.replace('.md', '.pdf')
    
    # Convert HTML to PDF
    pdfkit.from_string(html_content, pdf_file)
    print(f"PDF generated: {pdf_file}")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python3 markdown_to_pdf3.py <markdown_file>")
        sys.exit(1)
    
    markdown_file = sys.argv[1]
    markdown_to_pdf(markdown_file)
