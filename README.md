# GAME-Filter

<div id="top"></div>
<!--
*** README template is from: https://github.com/othneildrew/Best-README-Template
-->



<!-- PROJECT SHIELDS -->
<!--
*** I'm using markdown "reference style" links for readability.
*** Reference links are enclosed in brackets [ ] instead of parentheses ( ).
*** See the bottom of this document for the declaration of the reference variables
*** for contributors-url, forks-url, etc. This is an optional, concise syntax you may use.
*** https://www.markdownguide.org/basic-syntax/#reference-style-links
-->
 <!-- [![Contributors][contributors-shield]][contributors-url]
[![Forks][forks-shield]][forks-url]
[![Stargazers][stars-shield]][stars-url]
[![Issues][issues-shield]][issues-url]
  [![MIT License][license-shield]][license-url]
  [![LinkedIn][linkedin-shield]][linkedin-url] -->



<!-- PROJECT LOGO -->
<br />


  <h3 align="center">Software capable of identifying microlensing events detected with Gaia telescope and deriving the properties of lensing objects that cause them</h3>
<p align="left">Current collaborators: <a href="https://www2.ung.si/~tj0014/">Taj Jankovič</a>, <a href="https://www2.ung.si/~agomboc/ag/About_Me.html">Andreja Gomboc</a>, <a href="https://wyrzykow.wixsite.com/lukasz">Łukasz Wyrzykowsk </a>, Uroš Kostić, <a href="https://www2.ung.si/~tpetrushevska/">Tanja Petrushevska</a>, Mateusz Bronikowski  </p>
<p align="left">Former collaborators: Martina Larma, Mile Karlica</p>
  <p align="center">
    <br />
     <!-- <a href="https://github.com/othneildrew/Best-README-Template"><strong>Explore the docs »</strong></a>  -->
    <br />
    <br />
   <!-- <a href="https://github.com/othneildrew/Best-README-Template">View Demo</a>  -->

[Report a bug](https://github.com/tajjankovic/GAME-Filter/issues).

  
 <!--   <a href="issues">Request Feature</a>  -->
  </p>
</div>



<!-- TABLE OF CONTENTS -->
<details>
  <summary>Table of Contents</summary>
  <ol>
    <li>
      <a href="#about-the-project">About The Project</a>
      <ul>
               <li><a href="#prerequisites">Prerequisites</a></li>
               <li><a href="#installation">Installation</a></li>
      </ul>
    </li>
    <li><a href="#usage">Usage</a></li>
      <ul>
         <li><a href="#basic-steps">Basic Steps</a></li>
         <li><a href="#running-the-code">Running the code</a></li>
      </ul>
    </li>
   <!-- <li><a href="#roadmap">Roadmap</a></li> -->
  <!--   <li><a href="#contributing">Contributing</a></li> -->
  <!--   <li><a href="#license">License</a></li> -->
    <li><a href="#contact">Contact</a></li>
  </ol>
</details>



<!-- ABOUT THE PROJECT -->
## About The Project

<!-- [![Product Name Screen Shot][product-screenshot]](https://example.com)  -->

<p align="justify">  The Gaia mission's contribution to astrometry provides an unprecedented opportunity to
observe microlensing events. While photometric detection has its merits, we focus on the
identification of events through astrometric data. The aim is to leverage the astrometric
precision of Gaia to develop GAME Filter, a software capable of identifying such
microlensing events and to derive the properties of lensing objects that cause them. 


<p align="justify"> We have established a range and distributions of source and lens parameters and used
them to generate mock Gaia observations of microlensing events. Additionally, we have established a range and distributions of stellar binary system parameters and
used them to generate mock Gaia observations of binary events, which could potentially be
contaminants, i.e. interpreted as microlensing events.


<p>
    <a href="https://github.com/tajjankovic/GAME-Filter/">GAME Filter</a> is a software tool developed to identify microlensing events in the <i>Gaia</i> dataset and derive the properties of the lensing objects. The software reads <code>x<sub>obs</sub></code>, <code>x<sub>err</sub></code>, <code>&#916;x<sub>obs</sub></code>, <code>t<sub>obs</sub></code>, and <code>&#966;<sub>obs</sub></code> from the <i>Gaia</i> data files. GAME Filter calculates <code>x<sub>fit</sub></code>, the deviation along <code>&#966;<sub>obs</sub></code> at <code>t<sub>obs</sub></code>, for specific single source and microlensing parameters. The software then minimizes a scalar parameter
</p>
<p>
    <code>
        MUWE = &#40; &Sigma;<sub>i=1</sub><sup>N</sup> &#40;x<sub>obs,i</sub> - x<sub>fit,i</sub>&#41;<sup>2</sup> / x<sub>err,i</sub><sup>2</sup> &#41;<sup>1/2</sup>,
    </code>
</p>
<p>
    which indicates the goodness of the microlensing fit. <code>N</code> corresponds to the number of observations for a specific event. The minimization process utilizes the Limited-memory Broyden-Fletcher-Goldfarb-Shanno<a href="#fn1"><sup>1</sup></a> algorithm (L-BFGS-B) to explore the parameter space and determine the optimal single source and microlensing parameters for individual events. A more detailed description of the minimization process is provided in Appendix <a href="#app_game_filter">Appendix</a>.
</p>
<p>
    Following the minimization process, the minimizer might stop in an incorrect local minimum, failing to find the correct solution. Consequently, we establish criteria to determine when an event is recovered. These criteria are based on the value of MUWE after minimization <code>MUWE<sub>min</sub></code>, L2 optimality error <code>L<sub>opt</sub></code> (see <a href="#app_game_filter">Appendix</a>), initial guesses, and the boundaries imposed on individual parameters. We consider an event as recovered if the following criteria are met:
</p>
<ul>
    <li>0.9 &lt; <code>MUWE<sub>min</sub></code> &lt; 1.1.</li>
    <li><code>L<sub>opt</sub></code> &lt; 0.015<a href="#fn2"><sup>2</sup></a>.</li>
    <li>The values of <code>&#960;<sub>EE</sub></code>, <code>&#960;<sub>EN</sub></code>, and <code>u<sub>0</sub></code> differ from the initial guesses.</li>
    <li>The values of all parameters are within the imposed boundaries.</li>
</ul>

<p id="fn1"><sup>1</sup> We tested several minimization methods and found that the L-BFGS-B method produces the most accurate results. This is mainly due to its feature of constraining individual parameters within bounds.</p>

<p id="fn2"><sup>2</sup> We determine the critical value <code>L<sub>opt</sub> = 0.015</code> from <code>L<sub>opt</sub></code> histograms as the value where histograms show a sharp decline.</p>


<p align="right">(<a href="#top">back to top</a>)</p>






<!-- ### Built With -->

<!-- This section should list any major frameworks/libraries used to bootstrap your project. Leave any add-ons/plugins for the acknowledgements section. Here are a few examples. -->





### Prerequisites

* Python 3
* [astropy](https://www.astropy.org/)
* [astromet](https://github.com/zpenoyre/astromet.py)
* [scanning law](https://github.com/gaiaverse/scanninglaw)

### Installation



1. Download the Jupypter Notebook file
 
2. Install Python packages
   
<!-- * Instructions for installation on macOS Monterey 12:
   ```sh
   pip3 install matplotlib, pandas, healpy, pygeos, basemap

   ```-->
<!-- * Instructions for installation on Ubuntu 20.04:-->
<!-- * Instructions for installation on Windows 10:-->

                
<p align="right">(<a href="#top">back to top</a>)</p>



<!-- USAGE EXAMPLES -->
## Usage

### Basic steps

What does the code do:
* Generates Gaia mock observations for a range of single-source parameters (RA, DEC, pmra, pmdec, parallax) and saves them to .parquet files.
* Generates Gaia mock observations for a range of microlensing parameters (RA, DEC, pmra, pmdec, parallax, u0, thetaE, t0, tE, piEE, piEN) and saves them to .parquet files.
* Generates Gaia mock observations for a range of binary parameters (RA, DEC, pmra, pmdec, parallax, period, a, e, q, l, tperi, v_phi, v_omega, v_theta) and saves them to .parquet files.

     
<!--
### Running the code

* E.g. from the command line for $\Delta z=0.6$ and 1.2:
   ```sh
   python3.8 outflow.py --dz_list 0.6 1.2
   ```
-->



   

<p align="right">(<a href="#top">back to top</a>)</p>



<!-- ROADMAP-->

<!-- See the [open issues](https://github.com/othneildrew/Best-README-Template/issues) for a full list of proposed features (and known issues).




<!-- CONTRIBUTING 
## Contributing

Contributions are what make the open source community such an amazing place to learn, inspire, and create. Any contributions you make are **greatly appreciated**.

If you have a suggestion that would make this better, please fork the repo and create a pull request. You can also simply open an issue with the tag "enhancement".
Don't forget to give the project a star! Thanks again!

1. Fork the Project
2. Create your Feature Branch (`git checkout -b feature/AmazingFeature`)
3. Commit your Changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the Branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

<p align="right">(<a href="#top">back to top</a>)</p> -->



<!-- LICENSE 
## License

Distributed under the MIT License. See `LICENSE.txt` for more information.

<p align="right">(<a href="#top">back to top</a>)</p> -->



<!-- CONTACT -->
## Contact

Taj Jankovič - taj.jankovic@ung.si

Project Link: [[https://github.com/tajjankovic/GAME-Filter](https://github.com/tajjankovic/GAME-Filter)]

<p align="right">(<a href="#top">back to top</a>)</p>



<!-- ACKNOWLEDGMENTS -->
<!-- ## References
<a id="1">[1]</a> 
Bonnerot C., Lu W., 2020, Monthly Notices of the Royal Astronomical Society.

<a id="2">[2]</a> 
Bonnerot C., Lu W., 2020, Monthly Notices of the Royal Astronomical Society.

<a id="3">[3]</a> 
Bonnerot C., Lu W., 2020, Monthly Notices of the Royal Astronomical Society.

<a id="4">[4]</a> 
Bonnerot C., Lu W., 2020, Monthly Notices of the Royal Astronomical Society.

<a id="5">[5]</a> 
Bonnerot C., Lu W., 2020, Monthly Notices of the Royal Astronomical Society.

<a id="6">[6]</a> 
Bonnerot C., Lu W., 2020, Monthly Notices of the Royal Astronomical Society.

<a id="7">[7]</a> 
Bonnerot C., Lu W., 2020, Monthly Notices of the Royal Astronomical Society.

<a id="8">[8]</a> 
Bonnerot C., Lu W., 2020, Monthly Notices of the Royal Astronomical Society.

<a id="9">[9]</a> 
Bonnerot C., Lu W., 2020, Monthly Notices of the Royal Astronomical Society.

 -->


<!-- Use this space to list resources you find helpful and would like to give credit to. I've included a few of my favorites to kick things off!

* [Choose an Open Source License](https://choosealicense.com)
* [GitHub Emoji Cheat Sheet](https://www.webpagefx.com/tools/emoji-cheat-sheet) -->




<!-- MARKDOWN LINKS & IMAGES  -->
<!-- https://www.markdownguide.org/basic-syntax/#reference-style-links 
[contributors-shield]: https://img.shields.io/github/contributors/othneildrew/Best-README-Template.svg?style=for-the-badge

[license-shield]: https://img.shields.io/github/license/othneildrew/Best-README-Template.svg?style=for-the-badge
[license-url]: https://github.com/othneildrew/Best-README-Template/blob/master/LICENSE.txt
[linkedin-shield]: https://img.shields.io/badge/-LinkedIn-black.svg?style=for-the-badge&logo=linkedin&colorB=555
[linkedin-url]: https://linkedin.com/in/othneildrew
[product-screenshot]: images/screenshot.png
-->
