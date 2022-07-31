pyimpspec is a Python package that implements an application programming interface (API) for working with impedance spectra.
The target audience is researchers who use electrochemical impedance spectroscopy (EIS) though the program may also be useful in educational settings.
pyimpspec includes features such as:

- reading experimental data from multiple different file formats such as:
	- Eco Chemie data files (`.dfr`)
	- Gamry data files (`.dta`)
	- Ivium data files (`.idf` and `.ids`)
	- Microsoft Excel (`.xlsx`)
	- OpenDocument Spreadsheet (`.ods`)
	- plain-text files with character-separated values
- validation of impedance spectra by checking if the data is Kramers-Kronig transformable
- construction of equivalent circuits by parsing a circuit description code (CDC)
- equivalent circuit fitting as a means of extracting quantitative data
- simulation of impedance spectra
- basic visualization of impedance spectra using, e.g., Bode and Nyquist plots

Check out [DearEIS](https://vyrjana.github.io/DearEIS/), which is based on pyimpspec, if you would prefer to use a program with a graphical user interface (GUI).


## Recent news

<ul>
  {% for post in site.posts limit:5 %}
    <li>
      <a href="{{ site.baseurl }}{{ post.url }}">{{ post.title }} ({{ post.date | date: "%Y-%m-%d" }})</a>
    </li>
  {% endfor %}
</ul>

[Archive](archive.md)


## How to install

### Requirements

- [Python (version 3.6 or later)](https://www.python.org/)


### Supported platforms

- Linux
- Windows

pyimpspec may also work on other platforms (e.g., MacOS) but this has not yet been confirmed.


### Step-by-step

- Make sure to first install a recent version of Python if you do not already have it installed.
- Install pyimpspec using [pip](https://pip.pypa.io/en/stable/), which is the package installer for Python and should be included with most distributions of the official Python runtime.
Open a terminal (e.g., PowerShell on Windows) and execute the following command.

```
pip install pyimpspec
```


### Updating

Updating pyimpspec can be done by executing the following command:

```
pip install --upgrade pyimpspec
```


## How to use

The pyimpspec package should now be importable in, e.g., a Python script or a Jupyter notebook.
Check out [this Jupyter notebook](https://github.com/vyrjana/pyimpspec/blob/main/examples/examples.ipynb) for examples of how to use the various functions and classes.

[The API documentation can be found here](https://vyrjana.github.io/pyimpspec/api)


## Support

Bug reports, feature requests, and requests for help can be posted on [GitHub](https://github.com/vyrjana/pyimpspec/issues).


## Contributing

If you wish to help make pyimpspec even better, then please head on over to the [GitHub repository](https://github.com/vyrjana/pyimpspec) for more information in the README.md file.


## License

pyimpspec is licensed under the [GPLv3 or later](https://www.gnu.org/licenses/gpl-3.0.html).

The licenses of pyimpspec's dependencies and/or sources of portions of code are included in the [LICENSES folder on GitHub](https://github.com/vyrjana/pyimpspec/tree/main/LICENSES).
