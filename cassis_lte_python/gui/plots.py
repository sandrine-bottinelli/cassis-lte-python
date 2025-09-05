from __future__ import annotations

from cassis_lte_python.utils.logger import CassisLogger
from cassis_lte_python.utils.constants import COLOR_RESIDUAL
from cassis_lte_python.utils.settings import SETTINGS
# from cassis_lte_python.gui.basic_units import mhz, BasicUnit
import numpy as np
import matplotlib.pyplot as plt
import tkinter
# from tkinter import ttk
from matplotlib.backends.backend_tkagg import (
    FigureCanvasTkAgg, NavigationToolbar2Tk)
# Implement the default Matplotlib key bindings.
# from matplotlib.backend_bases import key_press_handler
# from matplotlib.figure import Figure
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib import ticker
import matplotlib
# matplotlib.use('Agg')

DPI_DEF = SETTINGS.DPI_DEF
NCOLS_DEF = SETTINGS.NCOLS_DEF
NROWS_DEF = SETTINGS.NROWS_DEF
FONT_DEF = SETTINGS.FONT_DEF

# Matplotlib global parameters
matplotlib.rcParams['xtick.direction'] = 'in'  # Ticks inside
matplotlib.rcParams['ytick.direction'] = 'in'  # Ticks inside
matplotlib.rcParams['ytick.right'] = True  # draw ticks on the right side
# axes.formatter.limits: -5, 6  # use scientific notation if log10
                               # of the axis range is smaller than the
                               # first or larger than the second
# axes.formatter.use_mathtext: False  # When True, use mathtext for scientific notation.
# axes.formatter.min_exponent: 0  # minimum exponent to format in scientific notation
matplotlib.rcParams['axes.formatter.useoffset'] = False  # No offset for tick labels
# axes.formatter.useoffset: True  # If True, the tick label formatter
                                 # will default to labeling ticks relative
                                 # to an offset when the data range is
                                 # small compared to the minimum absolute
                                 # value of the data.
# axes.formatter.offset_threshold: 4  # When useoffset is True, the offset
                                     # will be used when it can remove
                                     # at least this number of significant
                                     # digits from tick labels.
# matplotlib.rc('font', **{'family': 'sans-serif', 'sans-serif': ['Helvetica']})
# matplotlib.rc('font', **{'family': 'serif', 'serif': ['Times']})
plt.rcParams["font.family"] = FONT_DEF

# arbitrary plot width:
PLOT_WIDTH = 2.0  # inches

LOGGER = CassisLogger.create('plots')


def plot_window(lte_model, win, ax, ax2=None, number=True, auto=True, lw=1.0, axes_labels=True):
    """
    Plots a given window : overall model, individual components, line positions.
    :param lte_model: an object of class ModelSpectrum
    :param win: the Window to plot
    :param ax: the Axis on which to plot the Window
    :param ax2
    :param number: annotate the plot with the window's number at the bottom center
    :param auto: automatic ticks for top axis
    :param lw: linewidth in points
    :param axes_labels: whether to add labels for the axes
    :return:
    """

    label_left_pos = 0.025  # relative to the Axis
    label_bottom_pos = 0.075
    label_top_pos = 0.9
    label_right_pos = 0.975

    # ax.xaxis.set_major_locator(plt.MaxNLocator(4))
    ax.xaxis.set_minor_locator(ticker.AutoMinorLocator())
    ax.yaxis.set_minor_locator(ticker.AutoMinorLocator())

    if axes_labels:
        xlabel = 'Velocity' if win.bottom_unit == 'km/s' else 'Frequency'
        ax.set_xlabel(f'{xlabel} [{win.bottom_unit}]')
        ax.set_ylabel(f'Intensity [{lte_model.yunit}]')
        if win.bottom_unit != win.top_unit:
            ax2.set_xlabel(f'Frequency [{win.top_unit}]')

    # plot range used (or not) for chi2 calculation
    v_range = win.v_range_fit
    if v_range is not None and win.in_fit:
        ax.axvspan(v_range[0], v_range[1], facecolor='green', alpha=0.075)
    for f_range in win.f_ranges_nofit:
        ax.axvspan(f_range[0], f_range[1], facecolor='red', alpha=0.05)

    ymin_plot, ymax_plot = np.inf, -np.inf
    #  Plot components if more than one
    if (lte_model.minimize or lte_model.modeling) and (len(lte_model.cpt_list) > 1):
        for icpt, cpt in enumerate(lte_model.cpt_list):
            ax.step(win.x_mod_plot, win.y_mod_cpt[cpt.name], where='mid',
                    color=lte_model.cpt_cols[icpt], linewidth=lw)
            if len(win.y_mod_err_cpt) > 0:
                ax.fill_between(win.x_mod_plot, win.y_mod_cpt[cpt.name] - win.y_mod_err_cpt[cpt.name],
                                win.y_mod_cpt[cpt.name] + win.y_mod_err_cpt[cpt.name],
                                color=lte_model.cpt_cols[icpt], alpha=0.1)
                ymin_plot = min(ymin_plot, min(win.y_mod_cpt[cpt.name] - win.y_mod_err_cpt[cpt.name]))
                ymax_plot = max(ymin_plot, max(win.y_mod_cpt[cpt.name] + win.y_mod_err_cpt[cpt.name]))

    # Plot data and/or model
    if win.x_file is not None:  # data
        if win.y_res is not None:
            ax.step(win.x_file_plot, win.y_res, where='mid', color=COLOR_RESIDUAL, linewidth=lw)
        ax.step(win.x_file_plot, win.y_file, where='mid', color='k', linewidth=1.5 * lw)

    if lte_model.minimize or lte_model.modeling:  # model
        if win.x_file is not None:  # model on top of data -> red
            col = 'r'
            lw_m = 1.5 * lw
        else:  # model only -> black
            col = 'k'
            lw_m = lw
        ax.step(win.x_mod_plot, win.y_mod, where='mid', color=col, linewidth=lw_m)

    if win.y_mod_err is not None:
        ax.fill_between(win.x_mod_plot, win.y_mod - win.y_mod_err, win.y_mod + win.y_mod_err, color='red', alpha=0.1)
        ymin_plot = min(ymin_plot, min(win.y_mod - win.y_mod_err))
        ymax_plot = max(ymax_plot, max(win.y_mod + win.y_mod_err))

    # Define and set limits
    ax.relim()  # recompute the axis data limits -> does not work on fill_between???
    ymin, ymax = ax.get_ylim()
    ymin = min(ymin, ymin_plot)
    ymax = max(ymax, ymax_plot)
    ymin = ymin - 0.05 * (ymax - ymin)
    ymax = ymax + 0.1 * (ymax - ymin)
    if ymin == ymax:
        ymin -= 0.001  # arbitrary
        ymax += 0.001  # arbitrary
    ax.set_ylim(ymin, ymax)
    dy = ymax - ymin

    ax.set_xlim(win.bottom_lim)

    if ax2 is not None:
        ax2.set_xlim(win.top_lim)
        if not auto:
            nMajTicks = 3
            dfreqWinMHz = (max(win.top_lim) - min(win.top_lim))  # window size in MHz
            df = dfreqWinMHz / (nMajTicks + 1)  # separation between ticks in MHz
            if df >= 0.5:
                base = 1
                if df >= 5:
                    base = 5
            else:
                base = 5 / 10 ** np.ceil(abs(np.log10(df)) + 1)
            dfreqTickMHz = base * np.ceil(df / base)
            # print(dfreqWinMHz, df, dfreqTickMHz)
            # print(win.top_lim)

            steps = [1, 2, 3, 4, 5, 10]

            ax2.xaxis.set_major_locator(plt.MaxNLocator(nbins=nMajTicks+1, steps=steps))
            # ax2.xaxis.set_major_locator(ticker.MultipleLocator(dfreqTickMHz))
            # if (dfreqTickMHz % 5 == 0) and (dfreqTickMHz % 10 != 0):
            #     ax2.xaxis.set_minor_locator(ticker.AutoMinorLocator(5))

    # write transition number (left, bottom)
    if number and win.plot_nb > 0:
        ax.text(label_left_pos, label_bottom_pos, "{}".format(win.plot_nb),
                transform=ax.transAxes,  horizontalalignment='left',
                bbox=dict(facecolor='white', edgecolor='white', alpha=0.5),
                # fontsize='large',
                color=win.tag_colors[win.transition.tag])
                # color=lte_model.tag_colors[win.transition.tag])

    # plot all lines from modeled tags
    for icpt, lines_cpt in win.main_lines_display.items():
        # main lines : compute vertical positions, shifting down for each component
        y_pos = ymax - dy * np.array([0, 0.075]) - 0.025 * icpt * dy
        for irow, row in lines_cpt.iterrows():
            plot_line_position(ax, row.x_pos, y_pos, row.x_pos_err,
                               color=row.color, label=row.label, linewidth=lw * 1.5)

    # store labels and handles for main lines
    mainHandles, mainLabels = ax.get_legend_handles_labels()

    # other species :
    y_pos_other = ymin + (ymax - ymin) * np.array([0., 0.075])
    ls = '-'
    for irow, row in win.other_species_display.iterrows():
        plot_line_position(ax, row.x_pos, y_pos_other, row.x_pos_err,
                           color=row.color, label=row.label, linewidth=lw * 1.25, linestyle=ls)

    # store labels and handles for all lines and keep only those corresponding to the other species
    handles_all, labels_all = ax.get_legend_handles_labels()
    labels = labels_all[len(mainLabels):]
    handles = handles_all[len(mainHandles):]

    # keep unique labels
    newLabels, newHandles = [], []  # for main lines
    satLabels, satHandles = [], []  # for satellites lines
    for handle, label in zip(handles_all, labels_all):
        if label in mainLabels and label not in newLabels:
            newLabels.append(label)
            newHandles.append(handle)
        elif label in labels and label not in satLabels:
            satLabels.append(label)
            satHandles.append(handle)
    # upper left legend
    leg = ax.legend(newHandles, newLabels, labelcolor='linecolor', frameon=True,
                    # bbox_to_anchor=(xmin, y_pos[1] - 0.01 * (ymax - ymin)),
                    # bbox_transform=ax.transData,
                    bbox_to_anchor=(label_left_pos, label_top_pos),  # default : relative to the Axis
                    # bbox_transform=plt.gcf().transFigure,
                    loc='upper left',
                    alignment='left',
                    # fontsize='large',
                    labelspacing=0.15,  # vertical space between the legend entries, in font-size units (default: 0.5)
                    facecolor='white', edgecolor='white', framealpha=0.75,
                    borderpad=0.2,
                    handlelength=0, handletextpad=0, borderaxespad=0)
    # leg.get_frame().set_facecolor('white')
    for item in leg.legend_handles:
        item.set_visible(False)
    for text in leg.get_texts():
        col = win.tag_colors[text.get_text()]
        text.set_color(col)
        # text.set_weight('bold')

    # lower right legend
    sat_leg = ax.legend(satHandles, satLabels, labelcolor='linecolor', frameon=True,
                        # bbox_to_anchor=(xmax1 + padding * dx1, y_pos[1] - 0.02 * (ymax - ymin)),
                        # bbox_transform=ax.transData, loc='upper right',
                        # bbox_to_anchor=(xmax, y_pos_other[1] + 0.01 * (ymax - ymin)),
                        # bbox_transform=ax.transData,
                        bbox_to_anchor=(label_right_pos, label_bottom_pos),
                        loc='lower right', alignment='right',
                        # fontsize='small',
                        labelspacing=0.15,
                        facecolor='white', edgecolor='white', framealpha=0.75,
                        borderpad=0.2, handlelength=0, handletextpad=0, borderaxespad=0)
    for text in sat_leg.get_texts():
        text.set_fontstyle("italic")
        col = win.tag_colors[text.get_text()]
        # col = win.other_species_display[win.other_species_display.tag == text.get_text()].color.values[0]
        text.set_color(col)
        # text.set_weight('bold')

    for item in sat_leg.legend_handles:
        item.set_visible(False)

    # Manually add the first legend back
    ax.add_artist(leg)


def plot_line_position(x_axis, x_pos, y_range, x_pos_err, err_color=None, **kwargs):
    x_axis.plot([x_pos, x_pos], y_range, **kwargs)
    # plot error on line frequency
    if err_color is None:
        err_color = kwargs['color']
    x_axis.plot([x_pos - x_pos_err, x_pos + x_pos_err], 2 * [np.average(y_range)],
                color=err_color, linewidth=0.75)


def gui_plot(lte_model):
    if len(lte_model.cpt_list) > 1:
        color_message = []
        for i in range(len(lte_model.cpt_list)):
            color_message.append(f"{lte_model.cpt_list[i].name} - {lte_model.cpt_cols[i]}")
        LOGGER.info(f"Component colors are : {' ; '.join(color_message)}")

    fontsize = 16
    plt.rc('font', size=fontsize)
    plt.rc('axes', labelsize=fontsize)  # fontsize of the x and y labels
    plt.rc('xtick', labelsize=fontsize)  # fontsize of the x tick labels
    plt.rc('ytick', labelsize=fontsize)  # fontsize of the y tick labels

    nplots = len(lte_model.win_list_gui)

    root = tkinter.Tk()
    title = "LTEmodel"
    if lte_model.model_config.minimize:
        title += " - Results"
    root.wm_title(title)
    root.geometry("1000x700")
    # root.columnconfigure(0, weight=1)
    # root.columnconfigure(1, weight=3)
    # root.rowconfigure(0, weight=3)
    # root.rowconfigure(1, weight=1)

    fig, ax = plt.subplots(1, 1, figsize=(6, 4))

    ax2 = ax.twiny()
    # ax2.xaxis.set_major_locator(plt.MaxNLocator(4))
    ax2.xaxis.set_minor_locator(ticker.AutoMinorLocator())

    plot_window(lte_model, lte_model.win_list_gui[0], ax, ax2=ax2)
    canvas = FigureCanvasTkAgg(fig, master=root)  # A tk.DrawingArea.
    canvas.draw()

    # pack_toolbar=False will make it easier to use a layout manager later on.
    toolbar = NavigationToolbar2Tk(canvas, root, pack_toolbar=False)
    # navigation toolbar
    # toolbarFrame = tkinter.Frame(master=root)
    # toolbarFrame.grid(row=1, column=1)
    # toolbar = NavigationToolbar2Tk(canvas, toolbarFrame)

    toolbar.update()
    # toolbar.grid(row=1, column=1, sticky='ew')

    # canvas.mpl_connect(
    #     "key_press_event", lambda event: print(f"you pressed {event.key}"))
    # canvas.mpl_connect("key_press_event", key_press_handler)

    # Create a frame for the listbox+scrollbar, attached to the root window
    win_frame = tkinter.Frame(root)
    win_names = [win.name
                 for win in lte_model.win_list_gui] if nplots > 1 else [lte_model.win_list_gui[0].name[:-4]]
    len_max = 0
    for name in win_names:
        if len(name) > len_max:
            len_max = len(name)
    # Create a Listbox and attaching it to its frame
    var = tkinter.StringVar(root)
    var.set(win_names)
    win_list = tkinter.Listbox(win_frame, width=len_max, selectmode='single', activestyle='none',
                               listvariable=var)
    win_list.select_set(0)
    win_list.activate(0)
    win_list.focus_set()

    # Insert elements into the listbox
    # for values in range(100):
    #     win_list.insert(tkinter.END, values)

    # handle event
    def win_selected(event):
        """
        Handle item selected event for the windows' listbox
        """
        # get selected indices
        iwin = event.widget.curselection()[0]
        # clear axis
        ax.clear()
        plot_window(lte_model, lte_model.win_list_gui[iwin], ax, ax2=ax2)
        canvas.draw_idle()
        # canvas.flush_events()
        toolbar.update()

    def OnEntryUpDown(event):
        selection = event.widget.curselection()[0]

        if event.keysym == 'Up':
            selection = selection - 1 if selection > 0 else (event.widget.size() - 1)

        if event.keysym == 'Down':
            selection = selection + 1 if selection < (event.widget.size() - 1) else 0

        event.widget.selection_clear(0, tkinter.END)
        event.widget.select_set(selection)
        event.widget.activate(selection)
        event.widget.selection_anchor(selection)
        event.widget.see(selection)
        win_selected(event)

    win_list.bind('<<ListboxSelect>>', win_selected)
    win_list.bind("<Down>", OnEntryUpDown)
    win_list.bind("<Up>", OnEntryUpDown)

    # Create a Scrollbar attached to the listbox's frame
    win_scroll = tkinter.Scrollbar(win_frame, orient='vertical')
    # setting scrollbar command parameter to have a vertical view
    win_scroll.config(command=win_list.yview)
    # Attaching Listbox to Scrollbar
    # Since we need to have a vertical scroll we use yscrollcommand
    win_list.config(yscrollcommand=win_scroll.set)

    # button_quit = tkinter.Button(master=root, text="Quit", command=root.quit)
    # Packing order is important. Widgets are processed sequentially and if there
    # is no space left, because the window is too small, they are not displayed.
    # The canvas is rather flexible in its size, so we pack it last which makes
    # sure the UI controls are displayed as long as possible.

    # Add Listbox to the left side of its frame
    win_list.pack(side=tkinter.LEFT, fill=tkinter.BOTH)
    # Add Scrollbar to the right side
    win_scroll.pack(side=tkinter.RIGHT, fill='y')
    # Add list frame to root
    win_frame.pack(side="left", fill='y')
    # win_frame.grid(rowspan=2, column=0, sticky='ns')

    # button_quit.pack(side=tkinter.BOTTOM)
    toolbar.pack(side=tkinter.BOTTOM, fill=tkinter.X)
    canvas.get_tk_widget().pack(side=tkinter.RIGHT, fill=tkinter.BOTH, expand=1)
    # canvas.get_tk_widget().grid(row=0, column=1)

    tkinter.mainloop()


def file_plot(lte_model, filename, dirname=None, verbose=True,
              dpi=DPI_DEF, nrows=NROWS_DEF, ncols=NCOLS_DEF):
    """
    Produces a plot of the fit results. If several species, change page when plotting the next species.
    :param lte_model: an abject of class ModelSpectrum
    :param filename: name of output file.
    :param dirname: path to an output directory.
    :param verbose: if True, prints some information in the terminal, such as png file location
    :param dpi: the dpi value
    :param nrows: number of rows
    :param ncols: number of columns
    :return: None
    """

    plt.close()
    plt.ticklabel_format(style='plain')

    nplots = len(lte_model.win_list_file)
    if nplots == 0:
        raise IndexError("Nothing to plot.")

    if lte_model.model_config.bandwidth is None or lte_model.model_config.fit_full_range:
        win_per_sp = {'*': lte_model.win_list_file}
        bottom_label = "Frequency [MHz]"
    else:
        win_per_sp = {sp: [win for win in lte_model.win_list_file if sp in win.name] for sp in lte_model.tag_list}
        bottom_label = "Velocity [km/s]"
    win_per_sp = {k: v for k, v in win_per_sp.items() if len(v) > 0}

    if nplots == 1:
        nrows, ncols = 1, 1
    if nplots < (nrows * ncols):
        nrows = int(np.ceil(nplots / ncols))
        if nplots < ncols:
            nrows, ncols = 1, nplots

    # determine if more than one page
    if len(win_per_sp) == 1 and nplots <= (nrows * ncols):  # one page : keep user's extension
        file_path = lte_model.set_filepath(filename, dirname=dirname)
    else:
        file_path = lte_model.set_filepath(filename, dirname=dirname, ext='pdf')

    if verbose:
        LOGGER.info("Saving plot to {} \n    ...".format(file_path))

    margins = {
        'left': 0.5,  # left margin
        'right': 0.2,  # right margin
        'top': 0.4,  # top margin
        'bottom': 0.4  # bottom margin
    }

    if lte_model.model_config.bandwidth is None or lte_model.model_config.fit_full_range:
        # fill A4 in landscape
        fontsize = 30
        lw = 2
        fig_w = 0.9 * 29.7
        fig_h = 0.9 * 21
        plt.rcParams.update(
            {'font.size': fontsize})  # to change font size of all items : does not work for axes/ticks??
        # To change the size of specific items, use one or more of the following :
        plt.rc('font', size=fontsize)
        plt.rc('axes', labelsize=fontsize, linewidth=lw*2)  # fontsize of the x and y labels
        plt.rc(('xtick', 'ytick'), labelsize=fontsize)
        plt.rc(('xtick.major', 'ytick.major'), width=lw, size=10)

        fig, axes = plt.subplots(figsize=(fig_w, fig_h),
                                 dpi=dpi)
        plot_window(lte_model, list(win_per_sp.values())[0][0], ax=axes, lw=lw, axes_labels=True, auto=True)
        plt.tight_layout()
        # fig.subplots_adjust(left=0.1, right=0.9, bottom=0.1, top=0.95)
        plt.savefig(file_path)
        return

    else:
        fig_w, fig_h = file_fig_size(nrows, ncols, **margins)
        fontsize = 6  # round(20 / ncols)
        lw = 0.5

    plt.rcParams.update({'font.size': fontsize})  # to change font size of all items : does not work for axes/ticks??
    # To change the size of specific items, use one or more of the following :
    plt.rc('font', size=fontsize)
    plt.rc('axes', labelsize=fontsize)  # fontsize of the x and y labels
    plt.rc('xtick', labelsize=fontsize)  # fontsize of the x tick labels
    plt.rc('ytick', labelsize=fontsize)  # fontsize of the y tick labels

    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(fig_w, fig_h),
                             dpi=dpi, layout="constrained")
    for ax in fig.axes:
        # ax2 = None
        ax2 = ax.twiny()
        # ax.set_box_aspect(0.5)
        # ax2.set_box_aspect(0.5)
        # ax2.xaxis.set_major_locator(plt.MaxNLocator(4))
        ax2.xaxis.set_minor_locator(ticker.AutoMinorLocator())
        # ax2 = ax.secondary_xaxis('top',
        #                          functions=(velo2freq(win.transition.f_trans_mhz, lte_model.vlsr_file),
        #                                     freq2velo(win.transition.f_trans_mhz, lte_model.vlsr_file)))

    bbox = None
    # bbox = 'tight'

    with PdfPages(file_path) as pdf:
        for sp, win_list in win_per_sp.items():
            # compute the number of pages for the current species
            nplots = len(win_list)
            nb_pages = int(np.ceil(nplots / (nrows * ncols)))

            # Common labels
            fig.suptitle("Frequency [MHz]")
            fig.supxlabel(bottom_label)
            fig.supylabel(f'Intensity [{lte_model.yunit}]')

            for p in range(nb_pages):
                for i in range(nrows * ncols):
                    ax = fig.axes[i]
                    # ax2 = None
                    ax2 = fig.axes[nrows * ncols + i]
                    # try:  # takes longer than setting secondary axis before looping on the plots
                    #     ax2 = fig.axes[nrows * ncols + i]
                    # except IndexError:
                    #     ax2 = ax.twiny()
                    #     ax2.xaxis.set_minor_locator(ticker.AutoMinorLocator())

                    # Clear elements from the selected axis :
                    ax.clear()  # takes longer than clearing individual elements but
                    # clearing individual elements produces an error in some environments??? - TBC
                    # ax.lines.clear()
                    # ax.texts.clear()
                    # ax.patches.clear()

                    # Make sure the frame is visible
                    ax.set_frame_on(True)
                    ax2.set_frame_on(True)

                    plot_ind = p * nrows * ncols + i
                    # if global index greater than number of plots, also clear the axes themselves
                    if plot_ind >= nplots:
                        ax.set_frame_on(False)
                        ax.set_xticks([])
                        ax.set_yticks([])
                        # NB: could use ax.set_visible(False), but this changes the figure's layout -> ok?
                        if ax2 is not None:
                            ax2.set_frame_on(False)
                            ax2.set_xticks([])
                            ax2.set_yticks([])
                        continue

                    win = win_list[plot_ind]
                    plot_window(lte_model, win, ax=ax, ax2=ax2, lw=lw, axes_labels=False, auto=False)

                # Update position of common labels for the last page
                if p == (nb_pages - 1):
                    n_last = nplots - (nb_pages - 1) * nrows * ncols
                    nrows_last = n_last / ncols
                    nrows_last = int(np.ceil(nrows_last))
                    w_last, h_last = file_fig_size(nrows_last, ncols, **margins)

                    if nrows_last < nrows:
                        # move supylabel and supxlabel up
                        fig.texts[2].set_y((fig_h - h_last/2) / fig_h)  # supylabel ; works ok
                        # fig.texts[1].set_y((fig_h - h_last - margins['bottom']) / fig_h)  # supxlabel
                        # the above does not work??!! ; reset supxlabel and compute new ypos
                        fig.supxlabel("")
                        ypos_bottom_label = (fig_h - h_last - margins['bottom']) / fig_h
                        try:
                            fig.texts[3].set_y(ypos_bottom_label)
                        except IndexError:
                            fig.text(0.5, ypos_bottom_label, "Velocity [km/s]",
                                     fontsize=fig.texts[0].get_fontsize(), ha='center', va='bottom')
                    # else:
                    #     ypos_bottom_label = fig.texts[1].get_y()
                    if n_last < ncols:
                        # compute new xpos
                        w = fig.subplotpars.right - fig.subplotpars.left - (ncols - 1) * fig.subplotpars.hspace
                        w /= ncols
                        xpos = fig.subplotpars.left + (n_last * w + (n_last - 1) * fig.subplotpars.hspace) / 2
                        # move suptitle to xpos
                        fig.texts[0].set_x(xpos)
                        fig.texts[1].set_x(xpos)

    # fig.text(0.5, t+(1-t)*2/3, "Frequency [MHz]", ha='center', va='top')
    # fig.text(0.5, b/3, "Velocity [km/s]", ha='center', va='bottom')
    # fig.text(l/2, 0.5, "Intensity [K]", ha='center', va='center', rotation='vertical')
    # fig.tight_layout()
    # adjust : left, bottom, right, top are the positions of the edges of the subplots
    # as a fraction of figure width (l, r) or height (b, t) ;
    # w/hspace is the width/height of the padding between subplots as a fraction of avg Axes width/height
    # b = 0.125  # b_marg/fig_h
    # t = 0.875  # (b_marg + plot_h)/fig_h
    # l = 0.1  # l_marg/fig_w
    # r = 0.95  # (l_marg + plot_w)/fig_w
    # fig.subplots_adjust(bottom=b, top=t,
    #                     left=l, right=r,
    #                     wspace=wspace, hspace=hspace)

                pdf.savefig(fig, bbox_inches=bbox,
                            dpi=dpi)

        plt.close()
    #     # fig.savefig(file_path, bbox_inches='tight', dpi=dpi)

    if verbose:
        LOGGER.info("Done\n")


def file_fig_size(nrows, ncols, **kwargs):
    # width_a4 = 8.27  # 8.27 inches = 21 cm
    # height_a4 = 11.69  # 11.69 inches = 29.7 cm
    # aspect_a4 = width_a4 / height_a4  # height/width of a subplot
    aspect = 0.5

    sub_w = PLOT_WIDTH
    sub_h = sub_w * aspect

    wspace = 0.2  # space size in width
    hspace = wspace / aspect * 1.2  # space size in height
    l_marg = kwargs.get('left', 0.5)  # left margin
    r_marg = kwargs.get('right', 0.2)  # right margin
    t_marg = kwargs.get('top', 0.4)  # top margin
    b_marg = kwargs.get('bottom', 0.4)  # bottom margin

    fig_w_norm = ncols + (ncols - 1) * wspace
    fig_h_norm = nrows + (nrows - 1) * hspace  # + b_marg + t_marg
    plot_w = sub_w * fig_w_norm
    plot_h = sub_h * fig_h_norm
    fig_w = plot_w + l_marg + r_marg
    fig_h = plot_h + t_marg + b_marg

    return fig_w, fig_h
