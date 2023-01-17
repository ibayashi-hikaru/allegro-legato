from docutils import nodes
from sphinx.util.docutils import SphinxDirective
from docutils.nodes import Element, Node, list_item
from typing import Any, Dict, List
from sphinx import addnodes
from sphinx.util import logging
from sphinx.errors import SphinxError

class TableFromList(SphinxDirective):
    has_content = True
    required_arguments = 0
    optional_arguments = 0
    final_argument_whitespace = False
    option_spec = {
        'columns': int,
    }

    def run(self) -> List[Node]:
        ncolumns = self.options.get('columns', 2)
        node = addnodes.compact_paragraph()
        node.document = self.state.document
        self.state.nested_parse(self.content, self.content_offset, node)
        if len(node.children) != 1 or not isinstance(node.children[0],
                                                     nodes.bullet_list):
            reporter = self.state.document.reporter
            raise SphinxError('table_from_list content is not a list')
        fulllist = node.children[0]

        # fill list with empty items to have a number of entries
        # that is divisible by ncolumns
        if (len(fulllist) % ncolumns) != 0:
            missing = int(ncolumns - (len(fulllist) % ncolumns))
            for i in range(0,missing):
                fulllist += list_item()

        table = nodes.table()
        tgroup = nodes.tgroup(cols=ncolumns)
        table += tgroup

        for i in range(ncolumns):
            tgroup += nodes.colspec(colwidth=1)

        tbody = nodes.tbody()
        tgroup += tbody
        current_row = nodes.row()

        for idx, cell in enumerate(fulllist.children):
            if len(current_row.children) == ncolumns:
                tbody += current_row
                current_row = nodes.row()
            entry = nodes.entry()
            current_row += entry
            if len(cell.children) > 0:
                entry += cell.children[0]

        tbody += current_row
        return [table]

def setup(app):
    app.add_directive("table_from_list", TableFromList)
    return {
        'version': '0.1',
        'parallel_read_safe': True,
        'parallel_write_safe': True,
    }
