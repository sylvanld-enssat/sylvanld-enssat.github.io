from mkdocs.plugins import BasePlugin
import os, json, yaml

def debug(variable):
    print('='*80)
    print(variable)
    string = ""
    for name in dir(variable):
        string += "%s\t\t%s\n" % (name, getattr(variable, name))
    print(string)


class FrontMatter(BasePlugin):

    """def on_page_read_source(self, page, config):"""
        

    def on_page_read_source(self, page, config):
        with open(page.file.abs_src_path, 'r', encoding='utf-8-sig', errors='strict') as f:
            source = f.read()
        self.source = source
        return source

    def on_page_markdown(self, markdown, page, config, files):
        yaml_metadata = self.source.replace(markdown, '').replace('---', '').strip()
        metadata = yaml.safe_load(yaml_metadata)
        print(metadata)
        page.frontmatter = yaml.safe_load(yaml_metadata)
        return markdown

