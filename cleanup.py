import os
import glob
import tokenize
import io

print("Removing unused code and imports via autoflake...")
os.system("python -m autoflake --in-place --remove-all-unused-imports --remove-unused-variables --recursive .")

print("Removing comments from python files...")
def strip_comments(source):
    result = []
    g = tokenize.generate_tokens(io.StringIO(source).readline)
    try:
        for toknum, tokval, _, _, _ in g:
            if toknum != tokenize.COMMENT:
                result.append((toknum, tokval))
    except tokenize.TokenError:
        pass
    return tokenize.untokenize(result)

for filepath in glob.glob("*.py"):
    if filepath == "cleanup.py":
        continue
    with open(filepath, "r", encoding="utf-8") as f:
        source = f.read()
    
    clean = strip_comments(source)
    
    lines = clean.splitlines()
    final_lines = []
    blank_count = 0
    for line in lines:
        if line.strip() == "":
            blank_count += 1
            if blank_count <= 2:
                final_lines.append(line)
        else:
            blank_count = 0
            final_lines.append(line)
            
    with open(filepath, "w", encoding="utf-8") as f:
        f.write("\n".join(final_lines) + "\n")

print("Cleanup script successfully completed.")
