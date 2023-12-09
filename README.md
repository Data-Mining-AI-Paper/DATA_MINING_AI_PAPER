# ![title_2](https://github.com/Data-Mining-AI-Paper/DATA_MINING_AI_PAPER/assets/78012131/10d9f387-3088-4a6d-99a6-3daa8435c0cb)

> 2023 FALL Data Mining(SCE3313, F074) Project

## 🚩 Table of Contents

- [Project summary](#-Project-summary)
- [Project structure](#-Project-structure)
- [Requirements](#-Requirements)
- [Methods](#-Methods)
- [Results](#-Results)
- [Browser Support](#-browser-support)
- [Pull Request Steps](#-pull-request-steps)
- [Contributing](#-contributing)
- [TOAST UI Family](#-toast-ui-family)
- [Used By](#-used-by)
- [License](#-license)

## Project summary

### Analysis Challenges in NLP Papers: BOLT - Beyond Obstacles, Leap Together

- We want to solve the problem of accessing ACL papers in natural language processing research and proposes solutions.
- The methods that we tried include TF-IDF, SVD, and K-means clustering to derive insights from a dataset of 12,745 papers.
- Using the data that crawled the acl paper, we made below three outputs:
  1. Keyword trend analysis with graphs
  2. Word cloud by year
  3. Research topic trend with clusters by year

### Team member

| Dept     | Name          |
| -------- | ------------- |
| software | Kyunghyun Min |
| software | Jongho Baik   |
| software | Junseo Lee    |

## Project structure

### Directory

```bash
/DATA_MINING_AI_PAPER
├── output
│   ├── k-means
│   └── wordcloud
├── tempfiles
├── 1. Crawling ACL.ipynb
├── 2. preprocess.py
├── 3. k-mean_clustering_word2vect.py
├── 4. keyword_trend.py
├── 5. wordcloud_by_year.py
├── 6. topic_trend.py
├── tf_idf.py
├── tool.py
├── ACL_PAPERS.json
├── LICENSE
├── preprocessed_ACL_PAPERS.pickle
└── README.md
```

### Details

- `output/k-means`: Contain the results of k-means++ clustering, labels of clusters, info about instance of k.
- `output/wordcloud`: Contain wordclouds by year, from 1979 to 2023.
- `1. Crawling ACL.ipynb` and `2. preprocess.py`: Crawl papers and preprocessing data.
- `3. k-mean_clustering_word2vect.py`: Make clusters by k-means++.
- `4. keyword_trend.py`: Provide graphs about changes in the importance of keywords by year.
- `5. wordcloud_by_year.py`: Provide important keywords as wordclouds by year.
- `6. topic_trend.py`: Labeling the clusters made in `3. k-mean_clustering_word2vect.py`.

## Requirements

### Hardware Configuration

- Google Cloud Platform(GCP) Compute Engine VM N1 instance
  - Custom configuration:
    - Intel® Xeon® E5-2696V3 10vCPU
    - Ubuntu 23.10
    - 65GB RAM
    - 200GB storage

### Software Configuration

- Python 3.11.5
- Jupyter Notebook Dependencies:
  - IPython: 8.15.0
  - ipykernel: 6.25.0
  - ipywidgets: 8.0.4
  - jupyter_client: 7.4.9
  - jupyter_core: 5.3.0
  - jupyter_server: 1.23.4
  - jupyterlab: 3.6.3
  - nbclient: 0.5.13
  - nbconvert: 6.5.4
  - nbformat: 5.9.2
  - notebook: 6.5.4
  - qtconsole: 5.4.2
  - traitlets: 5.7.1

### Additional Libraries

To ensure consistency in package versions, the following additional libraries are used:

- matplotlib: 3.5.2
- numpy: 1.23.1
- nltk: 3.8.1
- scikit-learn: 1.3.0
- pyclustering: 0.10.1.2
- wordcloud: 1.9.2

## Method

## Result

## 📦 Packages

### TOAST UI Editor

| Name                                                                            | Description                |
| ------------------------------------------------------------------------------- | -------------------------- |
| [`@toast-ui/editor`](https://github.com/nhn/tui.editor/tree/master/apps/editor) | Plain JavaScript component |

### TOAST UI Editor's Wrappers

| Name                                                                                        | Description                                     |
| ------------------------------------------------------------------------------------------- | ----------------------------------------------- |
| [`@toast-ui/react-editor`](https://github.com/nhn/tui.editor/tree/master/apps/react-editor) | [React](https://reactjs.org/) wrapper component |
| [`@toast-ui/vue-editor`](https://github.com/nhn/tui.editor/tree/master/apps/vue-editor)     | [Vue](https://vuejs.org/) wrapper component     |

### TOAST UI Editor's Plugins

| Name                                                                                                                           | Description                     |
| ------------------------------------------------------------------------------------------------------------------------------ | ------------------------------- |
| [`@toast-ui/editor-plugin-chart`](https://github.com/nhn/tui.editor/tree/master/plugins/chart)                                 | Plugin to render chart          |
| [`@toast-ui/editor-plugin-code-syntax-highlight`](https://github.com/nhn/tui.editor/tree/master/plugins/code-syntax-highlight) | Plugin to highlight code syntax |
| [`@toast-ui/editor-plugin-color-syntax`](https://github.com/nhn/tui.editor/tree/master/plugins/color-syntax)                   | Plugin to color editing text    |
| [`@toast-ui/editor-plugin-table-merged-cell`](https://github.com/nhn/tui.editor/tree/master/plugins/table-merged-cell)         | Plugin to merge table columns   |
| [`@toast-ui/editor-plugin-uml`](https://github.com/nhn/tui.editor/tree/master/plugins/uml)                                     | Plugin to render UML            |

## 🤖 Why TOAST UI Editor?

TOAST UI Editor provides **Markdown mode** and **WYSIWYG mode**. Depending on the type of use you want like production of _Markdown_ or maybe to just edit the _Markdown_. The TOAST UI Editor can be helpful for both the usage. It offers **Markdown mode** and **WYSIWYG mode**, which can be switched any point in time.

### Productive Markdown Mode

![markdown](https://user-images.githubusercontent.com/37766175/121464762-71e2fc80-c9ef-11eb-9a0a-7b06e08d3ccb.png)

**CommonMark + GFM Specifications**

Today _CommonMark_ is the de-facto _Markdown_ standard. _GFM (GitHub Flavored Markdown)_ is another popular specification based on _CommonMark_ - maintained by _GitHub_, which is the _Markdown_ mostly used. TOAST UI Editor follows both [_CommonMark_](http://commonmark.org/) and [_GFM_](https://github.github.com/gfm/) specifications. Write documents with ease using productive tools provided by TOAST UI Editor and you can easily open the produced document wherever the specifications are supported.

- **Live Preview** : Edit Markdown while keeping an eye on the rendered HTML. Your edits will be applied immediately.
- **Scroll Sync** : Synchronous scrolling between Markdown and Preview. You don't need to scroll through each one separately.
- **Syntax Highlight** : You can check broken Markdown syntax immediately.

### Easy WYSIWYG Mode

![wysiwyg](https://user-images.githubusercontent.com/37766175/121808381-251f5000-cc93-11eb-8c47-4f5a809de2b3.png)

- **Table** : Through the context menu of the table, you can add or delete columns or rows of the table, and you can also arrange text in cells.
- **Custom Block Editor** : The custom block area can be edited through the internal editor.
- **Copy and Paste** : Paste anything from browser, screenshot, excel, powerpoint, etc.

### UI

- **Toolbar** : Through the toolbar, you can style or add elements to the document you are editing.
  ![UI](https://user-images.githubusercontent.com/37766175/121808231-767b0f80-cc92-11eb-82a0-433123746982.png)

- **Dark Theme** : You can use the dark theme.
  ![UI](https://user-images.githubusercontent.com/37766175/121808649-8136a400-cc94-11eb-8674-812e170ccab5.png)

### Use of Various Extended Functions - Plugins

![plugin](https://user-images.githubusercontent.com/37766175/121808323-d8d41000-cc92-11eb-9117-b92a435c9b43.png)

CommonMark and GFM are great, but we often need more abstraction. The TOAST UI Editor comes with powerful **Plugins** in compliance with the Markdown syntax.

**Five basic plugins** are provided as follows, and can be downloaded and used with npm.

- [**`chart`**](https://github.com/nhn/tui.editor/tree/master/plugins/chart) : A code block marked as a 'chart' will render [TOAST UI Chart](https://github.com/nhn/tui.chart).
- [**`code-syntax-highlight`**](https://github.com/nhn/tui.editor/tree/master/plugins/code-syntax-highlight) : Highlight the code block area corresponding to the language provided by [Prism.js](https://prismjs.com/).
- [**`color-syntax`**](https://github.com/nhn/tui.editor/tree/master/plugins/color-syntax) :
  Using [TOAST UI ColorPicker](https://github.com/nhn/tui.color-picker), you can change the color of the editing text with the GUI.
- [**`table-merged-cell`**](https://github.com/nhn/tui.editor/tree/master/plugins/table-merged-cell) :
  You can merge columns of the table header and body area.
- [**`uml`**](https://github.com/nhn/tui.editor/tree/master/plugins/uml) : A code block marked as an 'uml' will render [UML diagrams](http://plantuml.com/screenshot).

## 🎨 Features

- [Viewer](https://github.com/nhn/tui.editor/tree/master/docs/en/viewer.md) : Supports a mode to display only markdown data without an editing area.
- [Internationalization (i18n)](https://github.com/nhn/tui.editor/tree/master/docs/en/i18n.md) : Supports English, Dutch, Korean, Japanese, Chinese, Spanish, German, Russian, French, Ukrainian, Turkish, Finnish, Czech, Arabic, Polish, Galician, Swedish, Italian, Norwegian, Croatian + language and you can extend.
- [Widget](https://github.com/nhn/tui.editor/tree/master/docs/en/widget.md) : This feature allows you to configure the rules that replaces the string matching to a specific `RegExp` with the widget node.
- [Custom Block](https://github.com/nhn/tui.editor/tree/master/docs/en/custom-block.md) : Nodes not supported by Markdown can be defined through custom block. You can display the node what you want through writing the parsing logic with custom block.

## 🐾 Examples

- [Basic](https://nhn.github.io/tui.editor/latest/tutorial-example01-editor-basic)
- [Viewer](https://nhn.github.io/tui.editor/latest/tutorial-example04-viewer)
- [Using All Plugins](https://nhn.github.io/tui.editor/latest/tutorial-example12-editor-with-all-plugins)
- [Creating the User's Plugin](https://nhn.github.io/tui.editor/latest/tutorial-example13-creating-plugin)
- [Customizing the Toobar Buttons](https://nhn.github.io/tui.editor/latest/tutorial-example15-customizing-toolbar-buttons)
- [Internationalization (i18n)](https://nhn.github.io/tui.editor/latest/tutorial-example16-i18n)

Here are more [examples](https://nhn.github.io/tui.editor/latest/tutorial-example01-editor-basic) and play with TOAST UI Editor!

## 🌏 Browser Support

| <img src="https://user-images.githubusercontent.com/1215767/34348387-a2e64588-ea4d-11e7-8267-a43365103afe.png" alt="Chrome" width="16px" height="16px" /> Chrome | <img src="https://user-images.githubusercontent.com/1215767/34348590-250b3ca2-ea4f-11e7-9efb-da953359321f.png" alt="IE" width="16px" height="16px" /> Internet Explorer | <img src="https://user-images.githubusercontent.com/1215767/34348380-93e77ae8-ea4d-11e7-8696-9a989ddbbbf5.png" alt="Edge" width="16px" height="16px" /> Edge | <img src="https://user-images.githubusercontent.com/1215767/34348394-a981f892-ea4d-11e7-9156-d128d58386b9.png" alt="Safari" width="16px" height="16px" /> Safari | <img src="https://user-images.githubusercontent.com/1215767/34348383-9e7ed492-ea4d-11e7-910c-03b39d52f496.png" alt="Firefox" width="16px" height="16px" /> Firefox |
| :--------------------------------------------------------------------------------------------------------------------------------------------------------------: | :---------------------------------------------------------------------------------------------------------------------------------------------------------------------: | :----------------------------------------------------------------------------------------------------------------------------------------------------------: | :--------------------------------------------------------------------------------------------------------------------------------------------------------------: | :----------------------------------------------------------------------------------------------------------------------------------------------------------------: |
|                                                                               Yes                                                                                |                                                                                   11+                                                                                   |                                                                             Yes                                                                              |                                                                               Yes                                                                                |                                                                                Yes                                                                                 |

## 🔧 Pull Request Steps

TOAST UI products are open source, so you can create a pull request(PR) after you fix issues. Run npm scripts and develop yourself with the following process.

### Setup

Fork `main` branch into your personal repository. Clone it to local computer. Install node modules. Before starting development, you should check if there are any errors.

```sh
$ git clone https://github.com/{your-personal-repo}/tui.editor.git
$ npm install
$ npm run build toastmark
$ npm run test editor
```

> TOAST UI Editor uses [npm workspace](https://docs.npmjs.com/cli/v7/using-npm/workspaces/), so you need to set the environment based on [npm7](https://github.blog/2021-02-02-npm-7-is-now-generally-available/). If subversion is used, dependencies must be installed by moving direct paths per package.

### Develop

You can see your code reflected as soon as you save the code by running a server. Don't miss adding test cases and then make green rights.

#### Run snowpack-dev-server

[snowpack](https://www.snowpack.dev/) allows you to run a development server without bundling.

```sh
$ npm run serve editor
```

#### Run webpack-dev-server

If testing of legacy browsers is required, the development server can still be run using a [webpack](https://webpack.js.org/).

```sh
$ npm run serve:ie editor
```

#### Run test

```sh
$ npm test editor
```

### Pull Request

Before uploading your PR, run test one last time to check if there are any errors. If it has no errors, commit and then push it!

For more information on PR's steps, please see links in the Contributing section.

## 💬 Contributing

- [Code of Conduct](https://github.com/nhn/tui.editor/blob/master/CODE_OF_CONDUCT.md)
- [Contributing Guideline](https://github.com/nhn/tui.editor/blob/master/CONTRIBUTING.md)
- [Commit Convention](https://github.com/nhn/tui.editor/blob/master/docs/COMMIT_MESSAGE_CONVENTION.md)
- [Issue Guidelines](https://github.com/nhn/tui.editor/tree/master/.github/ISSUE_TEMPLATE)

## 🍞 TOAST UI Family

- [TOAST UI Calendar](https://github.com/nhn/tui.calendar)
- [TOAST UI Chart](https://github.com/nhn/tui.chart)
- [TOAST UI Grid](https://github.com/nhn/tui.grid)
- [TOAST UI Image Editor](https://github.com/nhn/tui.image-editor)
- [TOAST UI Components](https://github.com/nhn)

## 🚀 Used By

- [NHN Dooray! - Collaboration Service (Project, Messenger, Mail, Calendar, Drive, Wiki, Contacts)](https://dooray.com)
- [UNOTES - Visual Studio Code Extension](https://marketplace.visualstudio.com/items?itemName=ryanmcalister.Unotes)

## 📜 License

This software is licensed under the [MIT](https://github.com/nhn/tui.editor/blob/master/LICENSE) © 2023 Data-Mining-AI-Paper
