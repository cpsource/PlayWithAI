import markdown
import sys

def convert_markdown_to_html(md_file_path, output_file_path):
    # Read the Markdown content
    with open(md_file_path, 'r', encoding='utf-8') as md_file:
        md_content = md_file.read()

    # Convert Markdown to HTML
    html_content = markdown.markdown(md_content)

    # Write the HTML content to the output file
    with open(output_file_path, 'w', encoding='utf-8') as output_file:
        output_file.write(html_content)

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python script_name.py input_markdown.md output_html.html")
        sys.exit(1)

    md_file_path = sys.argv[1]
    output_file_path = sys.argv[2]

    convert_markdown_to_html(md_file_path, output_file_path)
    print(f"Converted {md_file_path} to {output_file_path}")

