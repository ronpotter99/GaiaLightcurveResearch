# This code retrieves TESS light curves for an object given its TIC number.
# It will combine data for all existing sectors and perform a period search
# using a Lomb-Scargle periodogram.
# It all searches for the object in Gaia DR2.
# The output is a plot showing the periodogram, the Gaia CMD, and the phase-
# folded light to both the period and twice the period (useful for binary
# systems where the dominant peak is often an alias).

__version__ = "3.0"
__author__ = "Ingrid Pelisoli and Thomas Stackhouse"
# Initial functionality provided by Ingrid Pelisoli
# Expanded functionality and refactor provided by Thomas Stackhouse

#####  IMPORTING PACKAGES  ######

from io import TextIOWrapper
from matplotlib.axes import Axes
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from astropy.io import ascii
from astropy.io import fits
from astropy.io.votable import parse_single_table
from astroquery.mast import Observations
from astroquery.gaia import Gaia
from astropy import wcs
from astropy.coordinates import SkyCoord, Distance, Angle
from astropy.time import Time
from astropy.visualization.mpl_normalize import ImageNormalize
import astropy.visualization as stretching
from lightkurve import search_targetpixelfile
import astropy.units as u
import sys
import pathlib
import TESSutils as tul


######################################
###  FUNCTIONS FOR GRAPH CREATION  ###
######################################


# Function that does the following:
# Takes two graph axes
# Downloads a target pixel file of the star being observed
# Calculates Gaia coordinates of the observed star
# Calculates what other stars may be in the area around the observed object
# Plots the target pixel file the first axes
# Plots an HR diagram on the second axes
# Returns two newly created graph axes
def gen_pixel_count_graph_and_gaia_hr_diagram(
    pixel_count_graph: Axes,
    gaia_hr_diagram: Axes,
    log_file: TextIOWrapper,
    crowd,
):
    print("Downloading pixel file for pixel count graph")
    tpf = search_targetpixelfile("TIC " + str(TIC), mission="TESS").download()
    
    # First do a large search using 6 pixels
    print("Calculating Gaia coordinates")

    coord = SkyCoord(
        ra=obsTable[0]["s_ra"],
        dec=obsTable[0]["s_dec"],
        unit=(u.degree, u.degree),
        frame="icrs",
    )
    # radius = u.Quantity(126.0, u.arcsec)
    rad = u.Quantity(126.0, u.arcsec)
    q = Gaia.cone_search_async(coord, radius=rad)
    gaia = q.get_results()
    # Select only those brighter than 18.
    gaia = gaia[gaia["phot_g_mean_mag"] < 18.0]
    gaia = gaia[np.nan_to_num(gaia["parallax"]) > 0]
    warning = len(gaia) == 0

    # Then propagate the Gaia coordinates to 2000, and find the best match to the
    # input coordinates
    if not warning:
        ra2015 = np.array(gaia["ra"]) * u.deg
        dec2015 = np.array(gaia["dec"]) * u.deg
        parallax = np.array(gaia["parallax"]) * u.mas
        pmra = np.array(gaia["pmra"]) * u.mas / u.yr
        pmdec = np.array(gaia["pmdec"]) * u.mas / u.yr
        c2015 = SkyCoord(
            ra=ra2015,
            dec=dec2015,
            distance=Distance(parallax=parallax, allow_negative=True),
            pm_ra_cosdec=pmra,
            pm_dec=pmdec,
            obstime=Time(2015.5, format="decimalyear"),
        )
        c2000 = c2015.apply_space_motion(dt=-15.5 * u.year)

        idx, sep, _ = coord.match_to_catalog_sky(c2000)

        # All objects
        id_all = gaia["SOURCE_ID"]
        plx_all = np.array(gaia["parallax"])
        g_all = np.array(gaia["phot_g_mean_mag"])
        MG_all = 5 + 5 * np.log10(plx_all / 1000) + g_all
        bprp_all = np.array(gaia["bp_rp"])

        id_all = np.array(id_all)
        g_all = np.array(gaia["phot_g_mean_mag"])
        MG_all = np.array(MG_all)
        bprp_all = np.array(bprp_all)

        # The best match object
        best = gaia[idx]
        gaia_id = best["SOURCE_ID"]

        MG = 5 + 5 * np.log10(best["parallax"] / 1000) + best["phot_g_mean_mag"]
        bprp = best["bp_rp"]

        gaia_id = int(gaia_id)
        G = float(best["phot_g_mean_mag"])
        MG = float(MG)
        bprp = float(bprp)

        # Coordinates for plotting
        radecs = np.vstack([c2000.ra, c2000.dec]).T
        coords = tpf.wcs.all_world2pix(radecs, 0.5)
        sizes = 128.0 / 2 ** ((g_all - best["phot_g_mean_mag"]))


        # Update log file with Gaia data
        print("Update log file with Gaia data")

        log_file.write("Gaia DR2 source_id = %20d\n" % (gaia_id))
        log_file.write("G = %6.3f, MG = %6.3f, bp_rp = %6.3f\n\n" % (G, MG, bprp))

        log_file.write("Other G < 18. matches within 5 pixels:\n")
        log_file.write("source_id            G       MG    bp_rp\n")
        for i in range(0, len(gaia)):
            if i != idx:
                log_file.write(
                    "%20d %6.3f %6.3f %6.3f\n"
                    % (id_all[i], g_all[i], MG_all[i], bprp_all[i])
                )

    else:
        log_file.write("Warning! No object with measured parallax within 30 arcsec.\n")

    # Reference sample

    table = parse_single_table("SampleC.vot")
    data = table.array

    s_MG = (
        5
        + 5 * np.log10(table.array["parallax"] / 1000)
        + table.array["phot_g_mean_mag"]
    )
    s_bprp = table.array["bp_rp"]


    # Generate pixel count graph
    print("Generating Pixel Count Graph")

    pixel_count_graph.set_title("TIC %d" % (TIC))
    mean_tpf = np.mean(tpf.flux.value, axis=0)
    nx, ny = np.shape(mean_tpf)
    norm = ImageNormalize(stretch=stretching.LogStretch())
    division = int(np.log10(np.nanmax(np.nanmean(tpf.flux.value, axis=0))))
    pixel_count_graph.imshow(
        np.nanmean(tpf.flux.value, axis=0) / 10**division,
        norm=norm,
        extent=[tpf.column, tpf.column + ny, tpf.row, tpf.row + nx],
        origin="lower",
        zorder=0,
    )
    pixel_count_graph.set_xlim(tpf.column, tpf.column + 10)
    pixel_count_graph.set_ylim(tpf.row, tpf.row + 10)
    if not warning:
        x = coords[:, 0] + tpf.column + 0.5
        y = coords[:, 1] + tpf.row + 0.5
        pixel_count_graph.scatter(x, y, c="firebrick", alpha=0.5, edgecolors="r", s=sizes)
        pixel_count_graph.scatter(x, y, c="None", edgecolors="r", s=sizes)
        pixel_count_graph.scatter(x[idx], y[idx], marker="x", c="white")
    pixel_count_graph.text(tpf.column, tpf.row, "crowdsap = %4.2f" % np.mean(crowd), color="w")
    pixel_count_graph.set_ylabel("Pixel count")
    pixel_count_graph.set_xlabel("Pixel count")


    # Generate Gaia HR Diagram
    print("Generating Gaia HR Diagram")

    gaia_hr_diagram.scatter(s_bprp, s_MG, c="0.75", s=0.5, zorder=0)
    if len(gaia) > 1:
        gaia_hr_diagram.scatter(bprp_all, MG_all, marker="s", c="b", s=10, zorder=1)
    gaia_hr_diagram.invert_yaxis()
    gaia_hr_diagram.set_title("$Gaia$ HR-diagram")
    if not warning:
        gaia_hr_diagram.plot(bprp, MG, "or", markersize=10, zorder=2)
    gaia_hr_diagram.set_ylabel("$M_G$")
    gaia_hr_diagram.set_xlabel("$G_{BP}-G_{RP}$")

    return pixel_count_graph, gaia_hr_diagram


# Function that does the following:
# Takes a graph axes
# Calculates the periodogram
# Plots power and period data on the axes
# Returns newly created graph axes
def gen_power_and_period_graph(
        power_and_period_graph: Axes,
        lcData: tul.LCdata,
):
    if not lcData.period:
        # Calculates the periodogram
        lcData.periodogram()
        if flag_ls == 1:
            print("logging periodogram data")
            ascii.write(
                [1 / lcData.freq, lcData.power],
                logs_dir + "TIC%09d_%d_ls.dat" % (TIC, lcData.exposure_time),
                names=["Period[h]", "Power"],
                overwrite=True,
            )

    # Power and Period Graph
    print("Generating Power and Period Graph")

    power_and_period_graph.set_title("Period = %5.2f h" % lcData.period)
    power_and_period_graph.plot(1.0 / lcData.freq, lcData.power, color="k")
    power_and_period_graph.set_xlim(min(1.0 / lcData.freq), max(1.0 / lcData.freq))
    power_and_period_graph.axhline(lcData.fap_001, color="b")
    power_and_period_graph.axvline(lcData.period, color="r", ls="--", zorder=0)
    # power_and_period_graph.axvspan(100., max(1.0/lcData.freq), alpha=0.5, color='red')
    power_and_period_graph.set_xscale("log")
    power_and_period_graph.set_xlabel("P [h]")
    power_and_period_graph.set_ylabel("Power")

    return power_and_period_graph


# Function that does the following:
# Takes a graph axes
# Plots bjd and flux information on the axes
# Returns newly created graph axes
def gen_bjd_graph(
    bjd_graph: Axes,
    bjd_original,
    flux_original,
    bjd_clean,
    flux_clean,
    sector_count,
    point_size=16,
):
    # Generating BJD graph
    print("Generating BJD graph")

    bjd_graph.set_title("%s sector/s" %(sector_count))
    bjd_graph.set_xlabel("BJD - 2457000")
    bjd_graph.set_ylabel("Relative flux")
    bjd_graph.set_xlim(np.min(bjd_clean), np.max(bjd_clean))
    # Plot the original values in light grey
    bjd_graph.scatter(bjd_original, flux_original, c="0.25", zorder=1, s=point_size)
    # Plot the cleaned values in black over original values
    bjd_graph.scatter(bjd_clean, flux_clean, c="k", zorder=1, s=point_size)

    return bjd_graph


# Function that does the following:
# Takes a graph axes
# Calculates periodogram if not already calculated
# Phases data using phase_factor
# Plots phased data on the axes
# Returns newly created graph axes
def gen_phase_to_dominant_peak_graph(
    phase_to_dominant_peak_graph: Axes,
    lc_data: tul.LCdata,
    log_file: TextIOWrapper,
    phase_factor,
    title = "Phased to dominant peak",
):
    if not lc_data.period:
        # Calculates the periodogram
        lc_data.periodogram()
        if flag_ls == 1:
            print("logging periodogram data")
            ascii.write(
                [1 / lc_data.freq, lc_data.power],
                logs_dir + "TIC%09d_%d_ls.dat" % (TIC, lc_data.exposure_time),
                names=["Period[h]", "Power"],
                overwrite=True,
            )

    # Generating Phase to Dominant Peak graph
    print("Generating Phase to Dominant Peak graph with phase factor %d" %(phase_factor))

    lc_data.phase_data(phase_factor)

    if flag_ph == 1:
        print("logging ascii file of phased data")
        ascii.write(
            [lc_data.phase, lc_data.flux_phased, lc_data.flux_err_phased],
            logs_dir + "TIC%09d_%d_phase_%d.dat" % (TIC, lc_data.exposure_time, phase_factor),
            names=["Phase", "RelativeFlux", "Error"],
            overwrite=True,
        )


    # Update log file with calculated phase data
    print("Update log file with calculated phase data")
    log_file.write(
        "Phase Factor = %d, Period = %9.5f hours, Amplitude =  %7.5f per cent\n"
        % (phase_factor, phase_factor * lc_data.period, 100 * abs(lc_data.amp))
    )
    log_file.write("FAP = %7.5e\n\n" % (lc_data.fap_p))


    # Phase to Dominant Peak Graph
    phi_avg = tul.avg_array(lc_data.phase, 50)
    fphi_avg = tul.avg_array(lc_data.flux_phased, 50)

    phase_to_dominant_peak_graph.set_title(title)
    phase_to_dominant_peak_graph.set_xlabel("Phase")
    phase_to_dominant_peak_graph.set_ylabel("Relative flux")
    phase_to_dominant_peak_graph.set_xlim(0, 2)
    # phase_to_dominant_peak_graph.errorbar(lc_data.phase, lc_data.flux_phased, fmt='.', color='0.5', markersize=0.75, elinewidth=0.5, zorder=0)
    phase_to_dominant_peak_graph.scatter(phi_avg, fphi_avg, marker=".", color="0.5", zorder=0)
    phase_to_dominant_peak_graph.plot(
        tul.running_mean(phi_avg, 15), tul.running_mean(fphi_avg, 15), ".k", zorder=1
    )
    phase_to_dominant_peak_graph.plot(lc_data.phase, lc_data.flux_fit, "r--", lw=3, zorder=2)
    # phase_to_dominant_peak_graph.errorbar(lc_data.phase + 1.0, lc_data.flux_phased, fmt='.', color='0.5', markersize=0.75, elinewidth=0.5, zorder=0)
    phase_to_dominant_peak_graph.scatter(phi_avg + 1.0, fphi_avg, marker=".", color="0.5", zorder=0)
    phase_to_dominant_peak_graph.plot(
        tul.running_mean(phi_avg, 15) + 1.0,
        tul.running_mean(fphi_avg, 15),
        ".k",
        zorder=1,
    )
    phase_to_dominant_peak_graph.plot(lc_data.phase + 1.0, lc_data.flux_fit, "r--", lw=3, zorder=2)


    return phase_to_dominant_peak_graph


################################################
###  FUNCTIONS FOR CREATING PAGES OF GRAPHS  ###
################################################


# This function makes an advanced plot with multiple subgraphs
def make_general_information_plot(
    lc_data: tul.LCdata,
    log_file: TextIOWrapper,
    bjd_original,
    flux_original,
    n_slow,
):
    plt.rcParams.update({"font.size": 22})
    fig = plt.figure(figsize=(24, 15), layout="tight")

    # A = Pixel count
    # B = Gaia HR diagram
    # C = Power/Period graph
    # D = flux/BJD graph
    # E = Phased to dominant peak
    # F = Phased to twice the peak
    subplotDict = fig.subplot_mosaic(
        """
        ABEEE
        ABEEE
        CCEEE
        CCFFF
        DDFFF
        DDFFF
        """,
    )

    # Generate TIC Pixel Count Graph and Gaia HR Diagrams
    pixel_count_graph = subplotDict.get('A')
    gaia_hr_diagram = subplotDict.get('B')
    gen_pixel_count_graph_and_gaia_hr_diagram(
        pixel_count_graph,
        gaia_hr_diagram,
        log_file,
        lc_data.crowdsap,
    )
    
    # Generate Power and Period Graph
    power_and_period_graph = subplotDict.get('C')
    gen_power_and_period_graph(
        power_and_period_graph,
        lc_data,
    )

    # Generate BJD Graph
    bjd_graph = subplotDict.get('D')
    gen_bjd_graph(
        bjd_graph,
        bjd_original,
        flux_original,
        lc_data.bjd,
        lc_data.flux,
        n_slow,
        point_size=0.5,
    )

    # Generate Phase to Dominant Peak Graph
    phase_to_dominant_peak_graph = subplotDict.get('E')
    gen_phase_to_dominant_peak_graph(
        phase_to_dominant_peak_graph,
        lc_data,
        log_file,
        1.0,
    )

    # Generate Phase to Twice Dominant Peak Graph
    phase_to_twice_dominant_peak_graph = subplotDict.get('F')
    gen_phase_to_dominant_peak_graph(
        phase_to_twice_dominant_peak_graph,
        lc_data,
        log_file,
        2.0,
        title = "Phased to twice the peak",
    )

    return fig


# This function makes a full-page BJD graph
def make_full_page_bjd_plot(
    lc_data: tul.LCdata,
    bjd_original,
    flux_original,
    sector_count,
    figsize=(48, 30),
    point_size=16,
    font_size=32,
):
    fig = plt.figure(figsize=figsize, layout="tight")
    subplot = fig.add_subplot()

    plt.rcParams.update({"font.size": font_size})

    gen_bjd_graph(
        subplot,
        bjd_original,
        flux_original,
        lc_data.bjd,
        lc_data.flux,
        sector_count,
        point_size,
    )

    return fig


# This function makes a split pixel count HR diagram graph
def make_full_page_location_plot(
    lc_data: tul.LCdata,
    log_file: TextIOWrapper,
    figsize=(48, 30),
    font_size=32,
):
    fig = plt.figure(figsize=figsize, layout="tight")
    subplots = fig.subplots(ncols=2)
    plt.rcParams.update({"font.size": font_size})

    gen_pixel_count_graph_and_gaia_hr_diagram(
        subplots[0],
        subplots[1],
        log_file,
        lc_data.crowdsap,
    )

    return fig


################################
#########  USER INPUT  #########
######  START OF SCRIPT  #######
################################


TIC_list = []
file_input_flag = False
create_detailed_graphs = False
skip_general_information_plot = False
show_questions_flag = False
question_input_flag = False
question_answers = None

# Process script arguments
for arg in sys.argv:

    # Check if argument is supposed to be an input file path
    if file_input_flag:

        # Read the input file
        with open(arg, "r") as file:
            for line in file:
                if line and line.strip().isdigit():
                    TIC_list.append(int(line.strip()))

        file_input_flag = False

    # Check if argument is supposed to be question inputs
    if question_input_flag:
        question_answers = arg
        question_input_flag = False

    # Check if argument is requesting help
    elif arg == "--help":
        print(
            """
            ########
            # HELP #
            ########

            Full usage:
            python3 TESS-LS.py [TIC #]... [OPTION]...


            To use this script, run the script name then any TIC number you would like to check:
            python3 TESS-LS.py [TIC #]

            Multiple TIC numbers may be provided to the script at once:
            python3 TESS-LS.py [TIC #] [TIC #]


            Additional options:

            --help
              Provides this help manual.

            -i
              The next argument will be the path of an input file. The input
              file should contain TIC numbers with each TIC number on a separate
              line. The easiest file format to use is '.txt'.
              Example:
              python3 TESS-LS.py [TIC #]... -i example_input.txt

            -q
              This flag provides questions for the user that can be used to
              gain additional information about the graphs or can modify some
              of the graph calculations. If this is used with the 'a' flag,
              the questions will not be displayed and the 'a' flag answers
              will be used instead.

            -a
              The next argument will be the values for user question answers.
              These questions are the same ones that show using the 'q' flag.
              There must be the same number of characters as there are questions.
              Example:
              python3 TESS-LS.py [TIC #]... -a 000

            -d
              Create detailed versions of the graphs. These are good for zooming in
              to get better resolution views of the data in case it is not clear if
              an outburst is spotted.

            -g
              Skip creating the general information plot. This makes the script run 
              much faster, however since some of the information normally required 
              in creating the general information plot is not calculated, the logs 
              will be sparser when running with this option.
"""
        )

    # Check if argument is a script flag
    elif len(arg) > 0 and arg[0] == "-":
        for flag in arg:

            # Check for 'i' flag to determine if next arg in list is an input file path
            # TIC numbers in the input file should be one TIC number per row
            if flag == "i":
                file_input_flag = True

            # Check for 'q' flag to determine if user questions should be shown
            # No flag means default answers are used
            # If used with the 'a' flag, that takes precedence
            if flag == "q":
                show_questions_flag = True

            # Check for 'a' flag to determine if next arg in list is question answers string
            if flag == "a":
                question_input_flag = True

            # Check for 'd' flag to determine if detailed graphs should be created
            if flag == "d":
                create_detailed_graphs = True

            # Check for 'g' flag to determine if general information plot should be skipped
            if flag == "g":
                skip_general_information_plot = True

    # Check if argument is a number
    elif arg.isdigit():
        TIC_list.append(int(arg))


if len(TIC_list) < 1:
    print("No TIC value provided. Exiting.")
    sys.exit()

# Output ascii light curve?
flag_lc = 0
# Output ascii periodogram?
flag_ls = 0
# Output ascii phase?
flag_ph = 0

if question_answers:
    if len(question_answers) == 3:
        flag_lc = int(question_answers[0])
        flag_ls = int(question_answers[1])
        flag_ph = int(question_answers[2])
    else:
        print(
            "There are 3 questions. If providing scripted answers, please answer all questions."
        )
        print("'%s' does not correctly answer the questions." % (question_answers))
        sys.exit()
elif show_questions_flag:
    flag_lc = int(
        input(
            "Would you like an ascii file of the processed light curve?\n0 = no (default), 1 = yes: "
        )
        or "0"
    )
    flag_ls = int(
        input(
            "Would you like an ascii file of the Lomb-Scargle periodogram, if calculated?\n0 = no (default), 1 = yes: "
        )
        or "0"
    )
    flag_ph = int(
        input(
            "Would you like an ascii file of the phased data, if calculated?\n0 = no (default), 1 = yes: "
        )
        or "0"
    )

for TIC in TIC_list:
    print(
        """

##################
# TIC %09d #
##################

          """
        % (TIC)
    )

    ################################
    #######  DOWNLOAD DATA  ########
    ################################

    # Searching for data at MAST

    obsTable = Observations.query_criteria(
        dataproduct_type="timeseries", project="TESS", target_name=TIC
    )

    # Create folder for results
    results_dir = "./Results_%09d/" % (TIC)
    print("Creating Results folder:", results_dir)
    pathlib.Path(results_dir).mkdir(exist_ok=True)

    # Create folder for additional logs
    logs_dir = results_dir + "logs/"
    print("Creating logs folder inside results folder")
    pathlib.Path(logs_dir).mkdir(exist_ok=True)

    # Download the 2-minute cadence light curves

    try:
        data = Observations.get_product_list(obsTable)
    except:
        log = open(results_dir + "TIC%09d_NO_DATA.log" % (TIC), "w")
        log.write("No data found for TIC %09d\n" % (TIC))
        log.close()
        continue

    download_lc = Observations.download_products(data, productSubGroupDescription="LC")
    infile = download_lc[0][:]
    sector_count_slow = len(infile)

    print(
        "I have found a total of %d 2-min light curve(s)."
        % (sector_count_slow)
    )

    # Download the 20-second cadence light curves

    download_fast_lc = Observations.download_products(
        data, productSubGroupDescription="FAST-LC"
    )

    if download_fast_lc is None:
        print("I have found no 20-sec light curves.")
        fast = False
    else:
        infile_fast = download_fast_lc[0][:]
        sector_count_fast = len(infile_fast)
        print(
            "I have found a total of %d 20-sec light curve(s)."
            % (sector_count_fast)
        )
        fast = True


    ################################
    #######  2-MINUTE DATA  ########
    ################################


    print("Processing 2-minute exposure data points")

    slow_lc = tul.LCdata(TIC, 120)

    slow_lc.read_data(infile)
    BJD_or = slow_lc.bjd
    flux_or = slow_lc.flux

    slow_lc.clean_data()
    if flag_lc == 1:
        print("logging ascii file of processed lightcurve")
        ascii.write(
            [slow_lc.bjd, slow_lc.flux, slow_lc.flux_err],
            logs_dir + "TIC%09d_%d_lc.dat" % (TIC, slow_lc.exposure_time),
            names=["BJD", "RelativeFlux", "Error"],
            overwrite=True,
        )

    print("Creating 2-minute exposure graphical images")

    # Set up slow log file
    slow_log = open(logs_dir + "TIC%09d.log" % (TIC), "w")
    slow_log.write("TIC %09d\n\n" % (TIC))
    slow_log.write("2-minute cadence data\n")
    slow_log.write("Number of sectors: %2d\n" % (sector_count_slow))
    slow_log.write("CROWDSAP: %5.3f\n" % (np.mean(slow_lc.crowdsap)))

    if not skip_general_information_plot:
        # Make general information graph
        plot = make_general_information_plot(
            slow_lc,
            slow_log,
            BJD_or,
            flux_or,
            sector_count_slow,
        )
        plot.savefig(results_dir + "TIC%09d.png" % (TIC))
    else:
        # Make a graph of the pixel count and HR diagram if the 
        # general information plot isn't generated. This helps
        # verify the white dwarf and any additional surrounding stars.
        # Since this doesn't change from fast to slow data, it doesn't
        # need to be generated for the fast data as well.
        location_graph = make_full_page_location_plot(
            slow_lc,
            slow_log,
        )
        location_graph.savefig(results_dir + "TIC%09d_pixel_count_graph_and_hr_diagram.png" % (TIC))

    # Make a bigger version of the slow BJD graph
    slow_full_graph = make_full_page_bjd_plot(
        slow_lc, BJD_or, flux_or, sector_count_slow
    )
    slow_full_graph.savefig(results_dir + "TIC%09d_full.png" % (TIC))

    # Make an expanded version of the slow BJD graph
    slow_expanded_graph = make_full_page_bjd_plot(
        slow_lc, BJD_or, flux_or, sector_count_slow, (120, 30)
    )
    slow_expanded_graph.savefig(results_dir + "TIC%09d_expanded.png" % (TIC))

    if create_detailed_graphs:
        # Make an expanded and detailed version of the slow BJD graph
        slow_expanded_detailed_graph = make_full_page_bjd_plot(
            slow_lc, BJD_or, flux_or, sector_count_slow, (120, 30), 2
        )
        slow_expanded_detailed_graph.savefig(
            results_dir + "TIC%09d_expanded_detailed.png" % (TIC)
        )

        # Make an extremely expanded and detailed version of the slow BJD graph
        slow_expanded_detailed_graph = make_full_page_bjd_plot(
            slow_lc, BJD_or, flux_or, sector_count_slow, (240, 30), 2
        )
        slow_expanded_detailed_graph.savefig(
            results_dir + "TIC%09d_extremely_expanded_detailed.png" % (TIC)
        )

    # Close log file
    slow_log.close()

    # Close generated graphs after they are saved so they do not stay open in memory
    plt.close('all')


    ################################
    ######  20-SECOND DATA  ########
    ################################


    if fast:
        print("Processing 20-second exposure data points")

        fast_lc = tul.LCdata(TIC, 20)

        fast_lc.read_data(infile_fast)
        BJD_or = fast_lc.bjd
        flux_or = fast_lc.flux

        fast_lc.clean_data()
        if flag_lc == 1:
            print("logging ascii file of processed lightcurve")
            ascii.write(
                [fast_lc.bjd, fast_lc.flux, fast_lc.flux_err],
                logs_dir + "TIC%09d_%d_lc.dat" % (TIC, fast_lc.exposure_time),
                names=["BJD", "RelativeFlux", "Error"],
                overwrite=True,
            )


        print("Creating 20-second exposure graphical images")

        # Set up fast log file
        fast_log = open(logs_dir + "TIC%09d_fast.log" % (TIC), "w")
        fast_log.write("TIC %09d\n\n" % (TIC))
        fast_log.write("20-second cadence data\n")
        fast_log.write("Number of sectors: %2d\n" % (sector_count_fast))
        fast_log.write("CROWDSAP: %5.3f\n" % (np.mean(fast_lc.crowdsap)))

        if not skip_general_information_plot:
            # Make general information graph
            plot_fast = make_general_information_plot(
                fast_lc,
                fast_log,
                BJD_or,
                flux_or,
                sector_count_fast,
            )
            plot_fast.savefig(results_dir + "TIC%09d_fast.png" % (TIC))

        # Make a bigger version of the fast BJD graph
        fast_full_graph = make_full_page_bjd_plot(
            fast_lc, BJD_or, flux_or, sector_count_fast
        )
        fast_full_graph.savefig(results_dir + "TIC%09d_fast_full.png" % (TIC))

        # Make an expanded version of the fast BJD graph
        fast_expanded_graph = make_full_page_bjd_plot(
            fast_lc, BJD_or, flux_or, sector_count_fast, (120, 30)
        )
        fast_expanded_graph.savefig(results_dir + "TIC%09d_fast_expanded.png" % (TIC))

        if create_detailed_graphs:
            # Make an expanded and detailed version of the fast BJD graph
            fast_expanded_detailed_graph = make_full_page_bjd_plot(
                fast_lc, BJD_or, flux_or, sector_count_fast, (120, 30), 2
            )
            fast_expanded_detailed_graph.savefig(
                results_dir + "TIC%09d_fast_expanded_detailed.png" % (TIC)
            )

            # Make an extremely expanded and detailed version of the fast BJD graph
            fast_expanded_detailed_graph = make_full_page_bjd_plot(
                fast_lc, BJD_or, flux_or, sector_count_fast, (240, 30), 2
            )
            fast_expanded_detailed_graph.savefig(
                results_dir + "TIC%09d_fast_extremely_expanded_detailed.png" % (TIC)
            )

        # Close log file
        fast_log.close()

        # Close generated graphs after they are saved so they do not stay open in memory
        plt.close('all')
