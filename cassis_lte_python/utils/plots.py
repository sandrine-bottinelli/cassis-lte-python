from cassis_lte_python.utils.utils import velocity_to_frequency, delta_v_to_delta_f
from cassis_lte_python.model_setup import select_transitions, get_transition_list, get_species_thresholds
from cassis_lte_python.model_setup import DATABASE_SQL, DPI_DEF
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tkinter
from tkinter import ttk
from matplotlib.backends.backend_tkagg import (
    FigureCanvasTkAgg, NavigationToolbar2Tk)
# Implement the default Matplotlib key bindings.
from matplotlib.backend_bases import key_press_handler
from matplotlib.figure import Figure
from matplotlib import ticker
import matplotlib
matplotlib.use('agg')


# CPT_COLORS = ['blue', 'green', 'mediumorchid']
# CPT_COLORS = [
#     'blue', 'dodgerblue', 'deepskyblue',
#     'orange',
#     'gold',
#     # 'yellow',
#     'green',
#     'purple', 'mediumorchid', 'pink']
TAB20 = plt.get_cmap('tab20')(np.linspace(0, 1, 20))
PLOT_COLORS = np.concatenate([TAB20[:][::2], TAB20[:][1::2]])
# PLOT_LINESTYLES = ['-', '--']
PLOT_LINESTYLES = ['-', ':']

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


def plot_window(lte_model, win, list_other_species=None, thresholds_other=None,
                other_species_selection=None, ax=None, fig=None, basic=False, dpi=None):
    """
    Plots a given window : overall model, individual components, line positions.
    :param lte_model: an abject of class ModelSpectrum
    :param win: the Window to plot
    :param list_other_species: file with list of other species (and their thresholds) to plot
    :param thresholds_other: TBC
    :param other_species_selection: deprecated
    :param ax: the Axis on which to plot the Window
    :param fig: the figure on which to plot the Window
    :param basic: plot only the overall model (no components, no line positions)
    :param dpi: desired dpi
    :return:
    """

    if fig is None:
        fig = Figure(figsize=(5, 4), dpi=dpi)
    if ax is None:
        ax = fig.add_subplot()
    if dpi is None:
        dpi = DPI_DEF

    vlsr = lte_model.cpt_list[0].vlsr if lte_model.vlsr_file == 0. else lte_model.vlsr_file
    best_pars = lte_model.best_params if lte_model.best_params is not None else lte_model.params2fit
    fwhm = max([best_pars[par].value for par in best_pars if 'fwhm' in par])

    tr = win.transition
    f_ref = tr.f_trans_mhz

    ax2 = ax.twiny()  # instantiate a second axes that shares the same y-axis
    padding = 0.05

    if lte_model.bandwidth is not None:  # velocity at bottom (1), freq at top (2)
        vmin, vmax = -lte_model.bandwidth / 2 + vlsr, lte_model.bandwidth / 2 + vlsr
        fmin, fmax = [velocity_to_frequency(v, f_ref, vref_kms=lte_model.vlsr_file)
                      for v in [vmax, vmin]]
        xmin1, xmax1 = vmin, vmax
        xmin2, xmax2 = fmin, fmax
        dx2 = xmax2 - xmin2
        dx1 = xmax1 - xmin1
        xlim1 = (xmin1 - padding * dx1, xmax1 + padding * dx1)
        xlim2 = (xmax2 + padding * dx2, xmin2 - padding * dx2)

        # Number of ticks
        ax2.xaxis.set_major_locator(plt.MaxNLocator(4))

    else:  # frequency at top and bottom
        fmin, fmax = lte_model.fmin_mhz, lte_model.fmax_mhz
        xmin1, xmax1 = fmin, fmax
        xmin2, xmax2 = fmin, fmax
        dx2 = xmax2 - xmin2
        dx1 = xmax1 - xmin1
        xlim1 = (xmin1 - padding * dx1, xmax1 + padding * dx1)
        xlim2 = xlim1

        # Number of ticks
        ax2.xaxis.set_major_locator(plt.MaxNLocator(4))
        ax.xaxis.set_major_locator(plt.MaxNLocator(4))

    ax2.set_xlim(xlim2)
    ax.set_xlim(xlim1)

    # Minor ticks
    ax2.xaxis.set_minor_locator(ticker.AutoMinorLocator())
    ax.xaxis.set_minor_locator(ticker.AutoMinorLocator())
    ax.yaxis.set_minor_locator(ticker.AutoMinorLocator())

    if lte_model.x_file is not None:
        fmin_mod = fmin
        fmax_mod = fmax
        x_file_win = lte_model.x_file[(fmin_mod <= lte_model.x_file) & (lte_model.x_file <= fmax_mod)]
        x_mod = np.linspace(min(x_file_win), max(x_file_win),
                            num=lte_model.oversampling * len(x_file_win))
        lte_model.x_mod = x_mod
    else:
        x_mod = lte_model.x_mod[(lte_model.x_mod <= fmax) & (lte_model.x_mod >= fmin)]
        y_mod = lte_model.y_mod[(lte_model.x_mod <= fmax) & (lte_model.x_mod >= fmin)]

    # compute model for all transitions (no thresholds)
    fwhm_mhz = delta_v_to_delta_f(fwhm, f_ref)
    all_lines = select_transitions(lte_model.line_list_all, xrange=[fmin - 2 * fwhm_mhz, fmax + 2 * fwhm_mhz], vlsr=vlsr)
    if lte_model.x_file is not None:
        y_mod = lte_model.compute_model_intensities(params=best_pars, x_values=x_mod, line_list=all_lines)
        lte_model.y_mod = y_mod

    all_lines_display = select_transitions(all_lines, xrange=[fmin, fmax], thresholds=lte_model.thresholds,
                                           vlsr=vlsr)  # for position display
    bright_lines = select_transitions(all_lines, xrange=[fmin, fmax], thresholds=lte_model.thresholds, vlsr=vlsr,
                                      bright_lines_only=True)
    other_lines_display = pd.concat([all_lines_display,
                                     bright_lines]).drop_duplicates(subset='db_id', keep=False)
    # other_lines_display = select_transitions(other_lines_display, xrange=[fmin, fmax])

    other_species_display = None
    if list_other_species is not None:
        try:
            other_lines_thresholds = get_transition_list(DATABASE_SQL, list_other_species, [[fmin, fmax]],
                                                         **thresholds_other, return_type='df')
            other_species_display = pd.concat([all_lines_display, bright_lines,
                                               other_lines_thresholds]).drop_duplicates(subset='db_id', keep=False)
        except IndexError:
            pass  # do nothing

    if lte_model.bandwidth is None:
        other_lines_display = None

    ymin = min(y_mod)
    ymax = max(y_mod)
    if lte_model.x_file is not None:
        x_file = lte_model.x_file[(fmin <= lte_model.x_file) & (lte_model.x_file <= fmax)]
        y_file = lte_model.y_file[(fmin <= lte_model.x_file) & (lte_model.x_file <= fmax)]
        ax2.plot(x_file, y_file, 'k-', drawstyle='steps-mid', linewidth=1)
        ymin = min(ymin, min(y_file))
        ymax = max(ymax, max(y_file))
    ymin = ymin - 0.05 * (ymax - ymin)
    ymax = ymax + 0.1 * (ymax - ymin)
    if ymin == ymax:
        ymin -= 0.001  # arbitrary
        ymax += 0.001  # arbitrary
    ax.set_ylim(ymin, ymax)

    # plot overall model
    ax2.plot(x_mod, y_mod, drawstyle='steps-mid', color='r', linewidth=1.5)

    # assign colors to tags
    tag_colors = {t: PLOT_COLORS[itag % len(PLOT_COLORS)] for itag, t in enumerate(lte_model.tag_list)}
    if list_other_species is not None:
        tag_other_sp_colors = {t: PLOT_COLORS[(itag + len(tag_colors)) % len(PLOT_COLORS)]
                               for itag, t in enumerate(list_other_species)}

    # write transition number (center, bottom)
    if len(lte_model.win_list_plot) > 1:
        ax2.text(0.5, 0.05, "{}".format(win.plot_nb),
                 transform=ax2.transAxes, horizontalalignment='center',
                 fontsize='large', color=tag_colors[tr.tag])

    # plot range used for chi2 calculation
    v_range = win.v_range_fit
    if v_range is not None:
        ax.axvspan(v_range[0], v_range[1], facecolor='purple', alpha=0.1)

    if not basic:  # plot line position(s) and plot components if more than one
        dy = ymax - ymin
        cpt_cols = plt.get_cmap('hsv')(np.linspace(0.1, 0.8, len(lte_model.cpt_list)))
        for icpt, cpt in enumerate(lte_model.cpt_list):
            vlsr_cpt = best_pars['{}_vlsr'.format(cpt.name)].value

            # plot line positions w/i user's constraints at the top
            all_lines_disp_cpt = all_lines_display[all_lines_display['tag'].isin(cpt.tag_list)]

            # compute vertical positions, shifting down for each component
            y_pos = ymax - dy * np.array([0, 0.075]) - 0.025 * icpt * dy

            for row in all_lines_disp_cpt.iterrows():
                tran = row[1].transition
                lbl = str(tran.tag)
                lw = 1.5
                plot_line_position(ax, tran, f_ref, vlsr_cpt, y_pos,
                                   color=tag_colors[tran.tag], label=lbl, linewidth=lw)

            # plot line positions outside user's constraints at the bottom
            other_lines_disp_cpt = None
            if other_lines_display is not None:
                other_lines_disp_cpt = other_lines_display[other_lines_display['tag'].isin(cpt.tag_list)]

            # compute vertical positions, shifting up for each component
            ypos_other = ymin + (ymax - ymin) * np.array([0., 0.075]) + 0.025 * (icpt + 1) * dy

            if other_lines_disp_cpt is not None:
                for row in other_lines_disp_cpt.iterrows():
                    tran = row[1].transition
                    lbl = "s{}".format(tran.tag)
                    lw = 0.75
                    col = tag_colors[tran.tag]
                    ypos_other = ymin + dy * np.array([0., 0.075])  # + 0.025 * dy
                    ls = ':'
                    plot_line_position(ax, tran, f_ref, vlsr_cpt, ypos_other,
                                       # linestyle=ls,
                                       color=col, label=lbl, linewidth=lw)

            if len(lte_model.cpt_list) > 1:
                c_y_mod = lte_model.compute_model_intensities(params=best_pars, x_values=x_mod, line_list=all_lines,
                                                              cpt=lte_model.cpt_list[icpt])

                ax2.plot(x_mod, c_y_mod, drawstyle='steps-mid',
                         color=cpt_cols[icpt % len(cpt_cols)], linewidth=0.5)

        if other_species_display is not None and len(other_species_display) >= 1:
            for row in other_species_display.iterrows():
                tran = row[1].transition
                lbl = "s{}".format(tran.tag)
                lw = 0.75
                col = tag_colors[tran.tag] if tran.tag in tag_colors.keys() else tag_other_sp_colors[tran.tag]
                ypos_other = ymin + (ymax - ymin) * np.array([0., 0.075])
                ls = '-'
                vlsr0 = best_pars['{}_vlsr'.format(lte_model.cpt_list[0].name)].value
                plot_line_position(ax, tran, f_ref, vlsr0, ypos_other, linestyle=ls,
                                   color=col, label=lbl, linewidth=lw)

        handles, labels = ax.get_legend_handles_labels()
        newLabels, newHandles = [], []  # for main lines
        satLabels, satHandles = [], []  # for satellites lines
        for handle, label in zip(handles, labels):
            if label not in newLabels and label[0] != 's':
                newLabels.append(label)
                newHandles.append(handle)
            elif label[1:] not in satLabels and label[0] == 's':
                satLabels.append(label[1:])
                satHandles.append(handle)
        leg = ax.legend(newHandles, newLabels, labelcolor='linecolor', frameon=False,
                        bbox_to_anchor=(xmin1 - padding * dx1, y_pos[1] - 0.01 * (ymax - ymin)),
                        bbox_transform=ax.transData, loc='upper left',
                        fontsize='small',
                        handlelength=0, handletextpad=0, fancybox=True)

        sat_leg = ax.legend(satHandles, satLabels, frameon=False, labelcolor='linecolor',
                            # bbox_to_anchor=(xmax1 + padding * dx1, y_pos[1] - 0.02 * (ymax - ymin)),
                            # bbox_transform=ax.transData, loc='upper right',
                            bbox_to_anchor=(xmax1 + padding * dx1, ypos_other[1] + 0.01 * (ymax - ymin)),
                            bbox_transform=ax.transData, loc='lower right',
                            fontsize='small',
                            handlelength=0, handletextpad=0, fancybox=True)
        for text in sat_leg.get_texts():
            text.set_fontstyle("italic")

        # Manually add the first legend back
        ax.add_artist(leg)

    return fig


def plot_line_position(x_axis, transition, freq_ref, v_cpt, y_range, err_color=None, **kwargs):
    # x_pos = velocity_to_frequency(vel, transition.f_trans_mhz, vref_kms=self.vlsr_file)
    x_pos = velocity_to_frequency(transition.f_trans_mhz, freq_ref, vref_kms=v_cpt, reverse=True)
    x_axis.plot([x_pos, x_pos], y_range, **kwargs)
    # plot error on line frequency
    if transition.f_err_mhz is not None:
        x_pos_err = delta_v_to_delta_f(transition.f_err_mhz, freq_ref, reverse=True)
        if err_color is None:
            err_color = kwargs['color']
        x_axis.plot([x_pos - x_pos_err, x_pos + x_pos_err], 2 * [np.average(y_range)],
                    color=err_color, linewidth=0.75)


def make_plot(lte_model, tag=None, filename=None, dirname=None, gui=False, verbose=True, basic=False,
              other_species=None, display_all=True, other_species_selection=None, dpi=None,
              pdf_multi=False):
    """
    Produces a plot of the fit results.
    :param lte_model: an abject of class ModelSpectrum
    :param tag: specify a tag to plot ; if None, all tags are plotted.
    :param filename: name of output file.
    :param dirname: path to an output directory.
    :param gui: if True, plot on screen, one window at a time.
    :param verbose: if True, prints some information in the terminal, such as png file location
    :param basic: if True, does not plot line position and individual components (time consuming)
    :param other_species: dictionary with key = tag, value = thresholds for other species
    for which to plot line position
    :param display_all: if False, display only lines with velocity selection.
    :param other_species_selection: (int) select only windows with other lines from this tag.
    :param dpi: the dpi value
    :return: None
    """

    plt.close()
    plt.ticklabel_format(style='plain')

    if dpi is None:
        dpi = DPI_DEF

    if tag is not None:
        win_list_plot = [w for w in lte_model.win_list if w.transition.tag == tag]
    else:
        win_list_plot = lte_model.win_list

    if not display_all:
        win_list_plot = [w for w in win_list_plot if w.in_fit]

    if other_species is not None:
        list_other_species, thresholds_other = get_species_thresholds(other_species)
    else:
        list_other_species, thresholds_other = None, None

    best_pars = lte_model.best_params if lte_model.best_params is not None else lte_model.params2fit
    lte_model.update_parameters(params=best_pars)

    if other_species_selection is None:
        lte_model.win_list_plot = win_list_plot
    else:
        fwhm_all_cpt = [best_pars[par].value for par in best_pars if 'fwhm' in par]
        fwhm = max(fwhm_all_cpt)
        for win in win_list_plot:
            f_ref = win.transition.f_trans_mhz
            # t_ref = win.transition.tag
            # fwhm_all_cpt = [best_pars[par].value for par in best_pars if 'fwhm_{}'.format(t_ref) in par]
            # fwhm = max(fwhm_all_cpt)
            delta_f = 3. * delta_v_to_delta_f(fwhm, fref_mhz=f_ref)
            try:
                res = get_transition_list(DATABASE_SQL, int(other_species_selection),
                                          [[f_ref - delta_f, f_ref + delta_f]],
                                          **thresholds_other, return_type='df')
                win.other_species_selection = res
                lte_model.win_list_plot.append(win)
            except IndexError:
                pass

    nplots = len(lte_model.win_list_plot)
    if nplots == 0:
        raise LookupError("No lines found for the plot. Please check your tag selection.")

    # if verbose:
    #     print("Tag {}, plot number {} :".format(tr.tag, line_list_plot.plot_nb.iloc[i]))
    #     print("Main transitions :")
    #     for t in all_lines_display['transition']:
    #         print(t)
    #     print("Satellite transitions :")
    #     for t in other_lines_display['transition']:
    #         print(t)
    #     print(" ")

    if gui:
        root = tkinter.Tk()
        root.wm_title("LTEmodel - Results")
        root.geometry("700x500")
        # root.columnconfigure(0, weight=1)
        # root.columnconfigure(1, weight=3)
        # root.rowconfigure(0, weight=3)
        # root.rowconfigure(1, weight=1)

        fig = plot_window(lte_model, lte_model.win_list_plot[0], list_other_species, thresholds_other,
                          other_species_selection)
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
        win_names = [win.name for win in lte_model.win_list_plot] if nplots > 1 else [lte_model.win_list_plot[0].name[:-4]]
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
            # update fig
            fig.clear()
            plot_window(lte_model, lte_model.win_list_plot[iwin], list_other_species, thresholds_other,
                        other_species_selection, fig=fig)
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

    if filename is not None:  # save to file
        file_path = lte_model.set_filepath(filename, dirname=dirname, ext='png')

        if verbose:
            print("\nSaving plot to {} \n...".format(file_path))

        nx = int(np.ceil(np.sqrt(nplots)))
        ny = int(np.ceil(nplots / nx))
        scale = 4
        fig, axes = plt.subplots(nx, ny, figsize=(nx * scale, ny * scale))
        for i, ax in enumerate(fig.axes):
            if i >= nplots:
                ax.set_frame_on(False)
                ax.set_xticks([])
                ax.set_yticks([])
                continue

            plot_window(lte_model, lte_model.win_list_plot[i],
                        list_other_species, thresholds_other, other_species_selection,
                        ax=ax, fig=fig, basic=basic, dpi=dpi)

        fig.savefig(file_path, bbox_inches='tight', dpi=dpi)

        if verbose:
            print("Done\n")
