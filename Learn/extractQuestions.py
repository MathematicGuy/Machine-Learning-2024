import re
import openpyxl

def extract_questions(txt_file, xlsx_file):
    with open(txt_file, 'r', encoding='utf-8') as f:
        content = f.read()

    questions = []
    # Biểu thức chính quy được cải tiến để xử lý xuống dòng
    matches = re.findall(r"Câu (\d+)\.\s*((?:.|\n)+?)(?=[A-D]\.|$)", content, re.MULTILINE)

    for match in matches:
        question_number = match[0]
        question_content = match[1].strip()
        options = re.findall(r"([A-D])\.\s*(.+)", question_content)
        options_dict = {option[0]: option[1].strip() for option in options}
        questions.append([question_number, question_content.replace('\n',' ').replace('A.','').replace('B.','').replace('C.','').replace('D.','').strip(), options_dict.get('A', ''), options_dict.get('B', ''), options_dict.get('C', ''), options_dict.get('D', '')])

    workbook = openpyxl.Workbook()
    sheet = workbook.active
    sheet.append(["Số thứ tự", "Nội dung câu hỏi", "A", "B", "C", "D"])
    for question in questions:
        sheet.append(question)

    workbook.save(xlsx_file)

# Sử dụng:
extract_questions("questions.txt", "output.xlsx")