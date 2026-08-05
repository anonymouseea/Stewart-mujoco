from pathlib import Path
from xml.sax.saxutils import escape
from zipfile import ZIP_DEFLATED, ZipFile
import csv
from datetime import datetime


class DataLogger:
    def __init__(self, path):
        self.path = Path(path)
        self.rows = []
        self.columns = []

    def record(self, **variables):
        row = {}
        for name, value in variables.items():
            flat = self._flatten(value)
            if len(flat) == 1:
                row[name] = flat[0]
                self._add_column(name)
            else:
                for i, item in enumerate(flat):
                    col = f"{name}_{i}"
                    row[col] = item
                    self._add_column(col)
        self.rows.append(row)

    def save(self, path=None):
        if path is not None:
            self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

        try:
            if self.path.suffix.lower() == ".csv":
                self._save_csv()
            else:
                self._save_xlsx()
        except PermissionError:
            self.path = self._timestamped_path(self.path)
            if self.path.suffix.lower() == ".csv":
                self._save_csv()
            else:
                self._save_xlsx()

    def _add_column(self, name):
        if name not in self.columns:
            self.columns.append(name)

    def _timestamped_path(self, path):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return path.with_name(f"{path.stem}_{timestamp}{path.suffix}")

    def _flatten(self, value):
        if hasattr(value, "flatten"):
            return value.flatten().tolist()
        if isinstance(value, (list, tuple)):
            out = []
            for item in value:
                out.extend(self._flatten(item))
            return out
        return [value]

    def _save_csv(self):
        with self.path.open("w", newline="", encoding="utf-8-sig") as f:
            writer = csv.DictWriter(f, fieldnames=self.columns)
            if self.columns:
                writer.writeheader()
            writer.writerows(self.rows)

    def _save_xlsx(self):
        def col_name(index):
            name = ""
            index += 1
            while index:
                index, rem = divmod(index - 1, 26)
                name = chr(65 + rem) + name
            return name

        def cell_xml(row_index, col_index, value):
            ref = f"{col_name(col_index)}{row_index}"
            if value is None:
                return f'<c r="{ref}"/>'
            if isinstance(value, str):
                return f'<c r="{ref}" t="inlineStr"><is><t>{escape(value)}</t></is></c>'
            return f'<c r="{ref}"><v>{float(value):.12g}</v></c>'

        sheet_rows = []
        all_rows = [self.columns] + [
            [row.get(column) for column in self.columns]
            for row in self.rows
        ]
        for row_index, row in enumerate(all_rows, start=1):
            cells = "".join(
                cell_xml(row_index, col_index, value)
                for col_index, value in enumerate(row)
            )
            sheet_rows.append(f'<row r="{row_index}">{cells}</row>')

        with ZipFile(self.path, "w", ZIP_DEFLATED) as workbook:
            workbook.writestr(
                "[Content_Types].xml",
                """<?xml version="1.0" encoding="UTF-8"?>
<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">
  <Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>
  <Default Extension="xml" ContentType="application/xml"/>
  <Override PartName="/xl/workbook.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet.main+xml"/>
  <Override PartName="/xl/worksheets/sheet1.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.worksheet+xml"/>
  <Override PartName="/xl/styles.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.styles+xml"/>
</Types>""",
            )
            workbook.writestr(
                "_rels/.rels",
                """<?xml version="1.0" encoding="UTF-8"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
  <Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="xl/workbook.xml"/>
</Relationships>""",
            )
            workbook.writestr(
                "xl/workbook.xml",
                """<?xml version="1.0" encoding="UTF-8"?>
<workbook xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main"
          xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships">
  <sheets><sheet name="data" sheetId="1" r:id="rId1"/></sheets>
</workbook>""",
            )
            workbook.writestr(
                "xl/_rels/workbook.xml.rels",
                """<?xml version="1.0" encoding="UTF-8"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
  <Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/worksheet" Target="worksheets/sheet1.xml"/>
  <Relationship Id="rId2" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/styles" Target="styles.xml"/>
</Relationships>""",
            )
            workbook.writestr(
                "xl/worksheets/sheet1.xml",
                f"""<?xml version="1.0" encoding="UTF-8"?>
<worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">
  <sheetData>{''.join(sheet_rows)}</sheetData>
</worksheet>""",
            )
            workbook.writestr(
                "xl/styles.xml",
                """<?xml version="1.0" encoding="UTF-8"?>
<styleSheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">
  <fonts count="1"><font><sz val="11"/><name val="Calibri"/></font></fonts>
  <fills count="1"><fill><patternFill patternType="none"/></fill></fills>
  <borders count="1"><border/></borders>
  <cellStyleXfs count="1"><xf numFmtId="0" fontId="0" fillId="0" borderId="0"/></cellStyleXfs>
  <cellXfs count="1"><xf numFmtId="0" fontId="0" fillId="0" borderId="0" xfId="0"/></cellXfs>
</styleSheet>""",
            )
