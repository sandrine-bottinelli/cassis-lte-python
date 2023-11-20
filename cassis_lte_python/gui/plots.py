from __future__ import annotations

from cassis_lte_python.utils.constants import COLOR_RESIDUAL
from cassis_lte_python.utils.settings import DPI_DEF
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
matplotlib.use('Agg')


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


def plot_window(lte_model, win, ax, ax2=None, number=True):
    """
    Plots a given window : overall model, individual components, line positions.
    :param lte_model: an object of class ModelSpectrum
    :param win: the Window to plot
    :param ax: the Axis on which to plot the Window
    :param ax2
    :param number: annotate the plot with the window's number at the bottom center
    :return:
    """

    label_left_pos = 0.025  # relative to the Axis
    label_bottom_pos = 0.075
    label_top_pos = 0.9
    label_right_pos = 0.975

    # ax.xaxis.set_major_locator(plt.MaxNLocator(4))
    ax.xaxis.set_minor_locator(ticker.AutoMinorLocator())
    ax.yaxis.set_minor_locator(ticker.AutoMinorLocator())

    # plot range used (or not) for chi2 calculation
    v_range = win.v_range_fit
    if v_range is not None:
        ax.axvspan(v_range[0], v_range[1], facecolor='green', alpha=0.1)
    for f_range in win.f_ranges_nofit:
        ax.axvspan(f_range[0], f_range[1], facecolor='red', alpha=0.1)

    #  Plot components if more than one
    if len(lte_model.cpt_list) > 1:
        for icpt, _ in enumerate(lte_model.cpt_list):
            ax.step(win.x_mod_plot, win.y_mod_cpt[icpt], where='mid',
                    color=lte_model.cpt_cols[icpt], linewidth=1)

    # Plot data and/or model
    if win.x_file is not None:  # data and model
        if win.y_res is not None:
            ax.step(win.x_file_plot, win.y_res, where='mid', color=COLOR_RESIDUAL, linewidth=1)
        ax.step(win.x_file_plot, win.y_file, where='mid', color='k', linewidth=1.5)
        ax.step(win.x_mod_plot, win.y_mod, where='mid', color='r', linewidth=1.5)
    else:  # model only
        ax.step(win.x_mod_plot, win.y_mod, where='mid', color='k', linewidth=1)

    # Define and set limits
    ax.relim()  # recompute the axis data limits
    ymin, ymax = ax.get_ylim()
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

    # write transition number (left, bottom)
    if number and win.plot_nb > 0:
        ax.text(label_left_pos, label_bottom_pos, "{}".format(win.plot_nb),
                transform=ax.transAxes,  horizontalalignment='left',
                bbox=dict(facecolor='white', edgecolor='white', alpha=0.5),
                fontsize='large', color=lte_model.tag_colors[win.transition.tag])

    # plot all lines from modeled tags
    for icpt, lines_cpt in win.main_lines_display.items():
        # main lines : compute vertical positions, shifting down for each component
        y_pos = ymax - dy * np.array([0, 0.075]) - 0.025 * icpt * dy
        lw = 1.5
        for irow, row in lines_cpt.iterrows():
            plot_line_position(ax, row.x_pos, y_pos, row.x_pos_err,
                               color=row.color, label=row.label, linewidth=lw)

    # store labels and handles for main lines
    mainHandles, mainLabels = ax.get_legend_handles_labels()

    # other species :
    y_pos_other = ymin + (ymax - ymin) * np.array([0., 0.075])
    lw = 1.5
    ls = '-'
    for irow, row in win.other_species_display.iterrows():
        plot_line_position(ax, row.x_pos, y_pos_other, row.x_pos_err,
                           color=row.color, label=row.label, linewidth=lw, linestyle=ls)

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
                    facecolor='white', edgecolor='white', framealpha=0.5,
                    borderpad=0.2,
                    handlelength=0, handletextpad=0, borderaxespad=0)
    # leg.get_frame().set_facecolor('white')
    for item in leg.legendHandles:
        item.set_visible(False)
    for text in leg.get_texts():
        col = win.tag_colors[text.get_text()]
        text.set_color(col)
        text.set_weight('bold')

    # lower right legend
    sat_leg = ax.legend(satHandles, satLabels, labelcolor='linecolor', frameon=True,
                        # bbox_to_anchor=(xmax1 + padding * dx1, y_pos[1] - 0.02 * (ymax - ymin)),
                        # bbox_transform=ax.transData, loc='upper right',
                        # bbox_to_anchor=(xmax, y_pos_other[1] + 0.01 * (ymax - ymin)),
                        # bbox_transform=ax.transData,
                        bbox_to_anchor=(label_right_pos, label_bottom_pos),
                        loc='lower right', alignment='right',
                        # fontsize='large',
                        facecolor='white', edgecolor='white', framealpha=0.5,
                        borderpad=0.2, handlelength=0, handletextpad=0, borderaxespad=0)
    for text in sat_leg.get_texts():
        text.set_fontstyle("italic")
        col = win.tag_colors[text.get_text()]
        # col = win.other_species_display[win.other_species_display.tag == text.get_text()].color.values[0]
        text.set_color(col)
        text.set_weight('bold')

    for item in sat_leg.legendHandles:
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
        print("Component colors are :", " ; ".join(color_message))

    fontsize = 16
    plt.rc('font', size=fontsize)
    plt.rc('axes', labelsize=fontsize)  # fontsize of the x and y labels
    plt.rc('xtick', labelsize=fontsize)  # fontsize of the x tick labels
    plt.rc('ytick', labelsize=fontsize)  # fontsize of the y tick labels

    nplots = len(lte_model.win_list_plot)

    root = tkinter.Tk()
    root.wm_title("LTEmodel - Results")
    root.geometry("1000x700")
    # root.columnconfigure(0, weight=1)
    # root.columnconfigure(1, weight=3)
    # root.rowconfigure(0, weight=3)
    # root.rowconfigure(1, weight=1)

    fig, ax = plt.subplots(1, 1, figsize=(5, 4))

    ax2 = ax.twiny()
    # ax2.xaxis.set_major_locator(plt.MaxNLocator(4))
    ax2.xaxis.set_minor_locator(ticker.AutoMinorLocator())

    plot_window(lte_model, lte_model.win_list_plot[0], ax, ax2=ax2)
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
                 for win in lte_model.win_list_plot] if nplots > 1 else [lte_model.win_list_plot[0].name[:-4]]
    # Create a Listbox and attaching it to its frame
    win_list = tkinter.Listbox(win_frame, width=10, selectmode='single', activestyle='none',
                               listvariable=tkinter.StringVar(value=win_names))
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
        plot_window(lte_model, lte_model.win_list_plot[iwin], ax, ax2=ax2)
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
              other_species=None, other_species_selection=None, dpi=None,
              nrows=4, ncols=3):
    """
    Produces a plot of the fit results.
    :param lte_model: an abject of class ModelSpectrum
    :param filename: name of output file.
    :param dirname: path to an output directory.
    :param verbose: if True, prints some information in the terminal, such as png file location
    :param other_species: dictionary with key = tag, value = thresholds for other species
    for which to plot line position
    :param other_species_selection: (int) select only windows with other lines from this tag.
    :param dpi: the dpi value
    :return: None
    """

    plt.close()
    plt.ticklabel_format(style='plain')
    fontsize = 12
    plt.rc('font', size=fontsize)
    plt.rc('axes', labelsize=fontsize)  # fontsize of the x and y labels
    plt.rc('xtick', labelsize=fontsize)  # fontsize of the x tick labels
    plt.rc('ytick', labelsize=fontsize)  # fontsize of the y tick labels

    if dpi is None:
        dpi = DPI_DEF

    nplots = len(lte_model.win_list_plot)

    # if verbose:
    #     print("Tag {}, plot number {} :".format(tr.tag, line_list_plot.plot_nb.iloc[i]))
    #     print("Main transitions :")
    #     for t in all_lines_display['transition']:
    #         print(t)
    #     print("Satellite transitions :")
    #     for t in other_lines_display['transition']:
    #         print(t)
    #     print(" ")

    if nplots == 0:
        raise IndexError("Nothing to plot.")

    # nplots = 13
    # nx = int(np.ceil(np.sqrt(nplots)))
    # ny = int(np.ceil(nplots / nx))
    if nplots == 1:
        nx, ny = 1, 1
    else:
        nx, ny = nrows, ncols
    # nx = 4
    # ny = 3

    # determine if more than one page
    nb_pages = int(np.ceil(nplots / (nx * ny)))
    if nb_pages == 1:  # one page : keep user's extension
        file_path = lte_model.set_filepath(filename, dirname=dirname)
    else:
        file_path = lte_model.set_filepath(filename, dirname=dirname, ext='pdf')

    if verbose:
        print("\nSaving plot to {} \n...".format(file_path))

    scale = 4
    fig, axes = plt.subplots(nx, ny, figsize=(nx * scale, ny * scale), dpi=dpi)
    axes2 = []

    # Draw first page
    for i in range(nx * ny):
        ax = fig.axes[i]
        if i >= nplots:
            ax.set_frame_on(False)
            ax.set_xticks([])
            ax.set_yticks([])
            # NB: could use ax.set_visible(False), but this changes the figure's layout -> ok?
            continue
        # ax2 = None
        ax2 = ax.twiny()
        # ax2.xaxis.set_major_locator(plt.MaxNLocator(4))
        ax2.xaxis.set_minor_locator(ticker.AutoMinorLocator())
        axes2.append(ax2)
        # ax2 = ax.secondary_xaxis('top',
        #                          functions=(velo2freq(win.transition.f_trans_mhz, lte_model.vlsr_file),
        #                                     freq2velo(win.transition.f_trans_mhz, lte_model.vlsr_file)))

        win = lte_model.win_list_plot[i]
        plot_window(lte_model, win, ax=ax, ax2=ax2)

    if nb_pages == 1:
        fig.savefig(file_path, bbox_inches='tight', dpi=dpi)

    else:
        with PdfPages(file_path) as pdf:
            # save first page
            pdf.savefig(fig)
            # plot and save pages 2+
            for p in range(1, nb_pages):
                for i in range(nx * ny):
                    ax = fig.axes[i]
                    ax2 = axes2[i]
                    # Clear elements from the selected axis :
                    ax.clear()  # takes longer than clearing individual elements but
                    # clearing individual elements produces an error in some environments??? - TBC
                    # ax.lines.clear()
                    # ax.texts.clear()
                    # ax.patches.clear()

                    plot_ind = p * nx * ny + i
                    if plot_ind >= nplots:
                        ax.set_frame_on(False)
                        ax.set_xticks([])
                        ax.set_yticks([])
                        if ax2 is not None:
                            ax2.set_frame_on(False)
                            ax2.set_xticks([])
                            ax2.set_yticks([])
                        continue

                    win = lte_model.win_list_plot[plot_ind]
                    plot_window(lte_model, win, ax=ax, ax2=ax2)

                pdf.savefig(fig)
        # last page
        # pdf.savefig(fig)
        plt.close()
    #     # fig.savefig(file_path, bbox_inches='tight', dpi=dpi)

    if verbose:
        print("Done\n")
