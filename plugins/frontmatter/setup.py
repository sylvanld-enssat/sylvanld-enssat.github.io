import setuptools

setuptools.setup(
    name='mkdocs-frontmatter',
    description='Enable loading of markdown front matter to use it templates.',
    version='0.0.1',
    packages=setuptools.find_packages(),

    entry_points={
        'mkdocs.plugins': [
            'frontmatter = frontmatter:FrontMatter',
        ]
    }
)