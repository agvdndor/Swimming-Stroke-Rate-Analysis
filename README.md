
<!-- PROJECT SHIELDS -->
<!--
*** I'm using markdown "reference style" links for readability.
*** Reference links are enclosed in brackets [ ] instead of parentheses ( ).
*** See the bottom of this document for the declaration of the reference variables
*** for contributors-url, forks-url, etc. This is an optional, concise syntax you may use.
*** https://www.markdownguide.org/basic-syntax/#reference-style-links
-->
[![Contributors][contributors-shield]][contributors-url]
[![Forks][forks-shield]][forks-url]
[![Stargazers][stars-shield]][stars-url]
[![Issues][issues-shield]][issues-url]
[![LinkedIn][linkedin-shield]][linkedin-url]



<!-- PROJECT LOGO -->
<br />
<p align="center">
  <a href="https://github.com/agvdndor/Swimming-Stroke-Rate-Analysis">
    <img src="assets/images/optical_vectors.png" alt="Logo" width="213" height="80">
  </a>

  <h3 align="center">Swimming Stroke Rate Analysis</h3>

  <p align="center">
    Automatically mine stroke rates from underwater video of swimmers over varying windows of time resolution.
    <br />
    <br />
    <a href="https://github.com/agvdndor/Swimming-Stroke-Rate-Analysis">Report Bug</a>
    ·
    <a href="https://github.com/agvdndor/Swimming-Stroke-Rate-Analysis">Request Feature</a>
  </p>
</p>



<!-- TABLE OF CONTENTS -->
## Table of Contents

* [About the Project](#about-the-project)
  * [Built With](#built-with)
* [Getting Started](#getting-started)
  * [Prerequisites](#prerequisites)
  * [Installation](#installation)
* [Usage](#usage)
* [Roadmap](#roadmap)
* [Contributing](#contributing)
* [License](#license)
* [Contact](#contact)
* [Acknowledgements](#acknowledgements)



<!-- ABOUT THE PROJECT -->
## About The Project

This project was realized as a thesis to obtain a Master of Engineering in Computer Science at Ghent University. For a comprehensive explanation of methods and results, please consult the [academic summary](assets/pdf/academic_summary.pdf).



### Abstract
Advances in the field of human pose estimation have significantly improved performance across complex datasets. However, current solutions that were designed and trained to recognize the human body across a wide range of contexts, e.g. MS COCO, often do not reach their full potential in very specific and challenging environments. This impedes subsequent analysis of the results. Underwater footage of competitive swimmers is an example of this, due to frequent self-occlusion of body parts and the presence of noise in the water. This work aims to improve the performance of pose estimation in this context in order to enable an automatic analysis of kinematics. Therefore, we propose a framework that limits the search space for human pose estimation by using a set of anchor poses. More specifically, the problem is reduced to finding the best matching anchor pose and the optimal transformation thereof.
	To find this best match, we devise a method of assessing similarity between two poses and use the Viterbi algorithm to find the most likely sequence of anchor poses. Thereby, we effectively exploit the cyclic character of the swimming motion.
	This does not only improve pose estimation performance but also provides a method to reliably extract the stroke frequency, outperforming manual timings by a human observer.

### Framework
The proposed framework consists of 3 main steps
- **Baseline Prediction:** Estimation of 13 keypoints by an existing human pose estimation model (preferrably finetuned on relevant dataset).
  <img src="assets/images/dataset_format.PNG">

- **Pose Matching:** Match the estimated pose to the most similar pose from a set of anchor poses. 

  <img src="assets/images/matching_outline.PNG">

- **Most Likely Sequence of Anchor Poses:** Use the Viterbi algorithm to obtain the most likely sequence of anchor poses given a series of consecutive pose predictions.

  <img src="assets/images/outline.png">
<!-- GETTING STARTED -->
## Getting Started

This is an example of how you may give instructions on setting up your project locally.
To get a local copy up and running follow these simple example steps.

### Prerequisites

This is an example of how to list things you need to use the software and how to install them.
* npm
```sh
npm install npm@latest -g
```

### Installation

1. Get a free API Key at [https://example.com](https://example.com)
2. Clone the repo
```sh
git clone https://github.com/your_username_/Project-Name.git
```
3. Install NPM packages
```sh
npm install
```
4. Enter your API in `config.js`
```JS
const API_KEY = 'ENTER YOUR API';
```


<!-- LICENSE -->
## License

Distributed under the MIT License. See `LICENSE` for more information.



<!-- CONTACT -->
## Contact

Your Name - [@your_twitter](https://twitter.com/your_username) - email@example.com

Project Link: [https://github.com/your_username/repo_name](https://github.com/your_username/repo_name)



<!-- ACKNOWLEDGEMENTS -->
## Acknowledgements
* [GitHub Emoji Cheat Sheet](https://www.webpagefx.com/tools/emoji-cheat-sheet)



<!-- MARKDOWN LINKS & IMAGES -->
<!-- https://www.markdownguide.org/basic-syntax/#reference-style-links -->
[contributors-shield]: https://img.shields.io/github/contributors/agvdndor/Swimming-Stroke-Rate-Analysis.svg?style=flat-square
[contributors-url]: https://github.com/agvdndor/Swimming-Stroke-Rate-Analysis/graphs/contributors
[forks-shield]: https://img.shields.io/github/forks/agvdndor/Swimming-Stroke-Rate-Analysis.svg?style=flat-square
[forks-url]: https://github.com/agvdndor/Swimming-Stroke-Rate-Analysis/network/members
[stars-shield]: https://img.shields.io/github/stars/agvdndor/Swimming-Stroke-Rate-Analysis.svg?style=flat-square
[stars-url]: https://github.com/agvdndor/Swimming-Stroke-Rate-Analysis/stargazers
[issues-shield]: https://img.shields.io/github/issues/othneildrew/Best-README-Template.svg?style=flat-square
[issues-url]: https://github.com/agvdndor/Swimming-Stroke-Rate-Analysis/issues
[license-shield]: https://img.shields.io/github/license/othneildrew/Best-README-Template.svg?style=flat-square
[linkedin-shield]: https://img.shields.io/badge/-LinkedIn-black.svg?style=flat-square&logo=linkedin&colorB=555
[linkedin-url]: https://www.linkedin.com/in/arne-vandendorpe-8800/
[product-screenshot]: images/screenshot.png
