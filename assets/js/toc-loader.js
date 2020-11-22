/**
 * Extract all div with class "toc" and append it an ul toc.
 */
(() => {
    function extractHeadings(containerId, maxLevel) {
        const content = document.getElementById(containerId);
        const levels = [...Array(maxLevel).keys()].map(level => 'h' + (level + 1)).join(',');
        console.log(levels);
        const headings = [...content.querySelectorAll(levels)].map(heading => ({
            link: '#' + heading.id,
            text: heading.textContent,
            level: Number(heading.tagName[1])
        }));
        return headings;
    }

    function toList(headings) {
        let string = "<ul>";
        let currentLevel = 0;
        for (let heading of headings) {
            if (heading.level > currentLevel) {
                for (k = currentLevel; k < heading.level; k++) {
                    string += `<li><ul>`;
                }
            } else if (heading.level < currentLevel) {
                for (k = heading.level; k < currentLevel; k++) {
                    string += `</ul></li>`;
                }
            }
            string += `<li><a href="${heading.link}">${heading.text}</a></li>`;
            currentLevel = heading.level;
        }
        string += "</ul>";
        return string;
    }

    window.injectTOC = (element) => {
        const headings = extractHeadings('content', 3);
        const tocInnerHTML = toList(headings);
        element.innerHTML = tocInnerHTML;
    }
})()