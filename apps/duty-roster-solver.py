import marimo

__generated_with = "0.14.7"
app = marimo.App(width="medium")


@app.cell
def _():
    # ## 1. Imports and Configuration
    import marimo as mo
    import pandas as pd
    import io
    from pathlib import Path
    from ortools.sat.python import cp_model
    from datetime import timedelta, date
    from collections import defaultdict
    return Path, cp_model, date, defaultdict, io, mo, pd, timedelta


@app.cell
def _(mo):
    mo.md(
        r"""
    /// details | Ambulance Roster Scheduling Solver

    # **Solver Objective**
        To generate a fair and optimal weekly work roster for a fleet of ambulances that strictly adheres to all operational demands and work rules, while strongly optimizing for fairness and shift consistency.

    ## **Hard Requirements**

        *   **R1: Exactly Meet Demand**
            The number of scheduled ambulances for each shift (`H`, `N`, `Dv`, etc.) must be *exactly equal* to the DAA specified in the `INPUT` sheet.

        *   **R2: One Shift Per Day**
            Each ambulance must be assigned exactly one shift per day (this includes rest shifts like 'O').

        *   **R3: Night Shift Rest**
            An ambulance that works a night shift (`Type` 'N') *must* be assigned an "Off" shift ('O') on the immediately following day.

        *   **R4: 48-Hour Work Week**
            Each ambulance must work **exactly 48 hours** per week (Monday to Sunday). The solver is allowed to use the flexibility of `Flexible` shifts (adjusting their hours within the `Working_Hour_Delta`) to achieve this target.

    ## **Soft Goals**

        *   **R5: Fairness of Tough Shifts**
            The total number of "tough" shifts ('H' and 'N') should be distributed as evenly as possible among all ambulances. The solver will strongly penalize imbalance, but perfect equality is not strictly required.

        *   **R6: Consistency of Day Shifts**
            The solver should try to assign as few different types of "D" shifts as possible to the same ambulance. The importance of this goal is controlled by the "D-Shift Consistency Priority" slider in the user interface.

        *   **R7: Consistency of Night Shifts**
            The solver should try to assign as few different types of "N" shifts as possible to the same ambulance. The importance of this goal is controlled by the "N-Shift Consistency Priority" slider in the user interface.

    ///
    """
    )
    return


@app.cell
def _(Path, mo):
    # Global UI elements for cross-cell reference
    FILE_DIR = "apps/duty-roster-solver-files"

    file_browser = mo.ui.file_browser(
        initial_path=Path(FILE_DIR),
        filetypes=[".xlsx"],
        restrict_navigation=True,
        multiple=False,
        label="Select an existing roster file:"
    )

    file_uploader = mo.ui.file(
        filetypes=[".xlsx"],
        label="Or upload your own roster file (.xlsx):"
    )

    file_input_tabs = mo.ui.tabs({
        "Choose Existing": file_browser,
        "Upload New": file_uploader,
    })

    file_input_tabs

    return FILE_DIR, file_browser, file_uploader


@app.cell
def _(FILE_DIR, Path, file_browser, file_uploader, io, mo, pd):
    # This cell depends on global file_browser and file_uploader from Cell 2
    _excel_file = None
    df_input = None
    df_shifts = None
    col_date = None

    if file_uploader.value:
        # Save uploaded file to target directory
        _target_dir = Path(FILE_DIR)
        _target_dir.mkdir(parents=True, exist_ok=True)
        _target_path = _target_dir / file_uploader.name()
        with open(_target_path, "wb") as f:
            f.write(file_uploader.contents())

        _excel_file = pd.ExcelFile(io.BytesIO(file_uploader.contents()))
        _file_status = mo.md(f"🟢 Using uploaded file: **{file_uploader.name()}**")
    elif file_browser.value:
        _excel_file = pd.ExcelFile(file_browser.path())
        _file_status = mo.md(f"🟡 Using selected file: **{file_browser.path().name}**")
    else:
        _file_status = mo.md("⚠️ **Please select an existing file or upload your own roster file (.xlsx) to proceed!**")

    if _excel_file is not None:
        # Read and process data
        df_input = pd.read_excel(_excel_file, sheet_name='INPUT')
        col_date = df_input.columns[0]
        df_input[col_date] = pd.to_datetime(df_input[col_date]).dt.date

        df_shifts = pd.read_excel(_excel_file, sheet_name='SHIFTS')

        _tabs = mo.ui.tabs({
            "INPUT Sheet": mo.ui.table(df_input),
            "SHIFTS Sheet": mo.ui.table(df_shifts)
        })
        # Display status and data
        data_display = mo.vstack([
            _file_status,
            _tabs
        ])
    else:
        data_display = _file_status

    data_display

    return col_date, df_input, df_shifts


@app.cell
def _(df_input, df_shifts, mo):
    input_error = False
    if df_input is not None and df_shifts is not None:
        input_shifts = set(df_input['Shift'].unique())
        shifts_shifts = set(df_shifts['Shift'].unique())
        invalid_shifts = input_shifts - shifts_shifts

        if invalid_shifts:
            input_error = True
            validation_msg = mo.md(
                f"❌ **Validation Error:** The following shifts in INPUT are not defined in the SHIFTS sheet: "
                f"`{', '.join(sorted(invalid_shifts))}`"
            )
        else:
            validation_msg = mo.md("✅ All shift codes in INPUT are valid and defined in the SHIFTS sheet.")
    else:
        validation_msg = mo.md("*Load both INPUT and SHIFTS sheets to validate shift codes.*")

    validation_msg

    return (input_error,)


@app.cell
def _(df_input, df_shifts, input_error, mo):
    # cell 4: Solver Settings
    # Define the state variable to track if the solver is running.
    get_solving, set_solving = mo.state(False)

    number_of_amb_input = mo.ui.number(
        start=1, stop=100, step=1, value=48,  # Default: 48 ambulances
        label="Total ambulances available:"
    )

    # Division selector (only if present)
    if df_input is not None and "Division" in df_input.columns:
        division_values = sorted(df_input["Division"].dropna().unique())
        division_selector = mo.ui.dropdown(
            options=division_values,
            value=division_values[0] if division_values else None,
            label="Select Division:"
        )
    else:
        division_selector = None

    # The run_button is disabled when get_solving() is True.
    run_button = mo.ui.run_button(
        label="🚀 Generate Schedule",
        disabled=get_solving()
    )

    # --- Priority Sliders for Soft Constraints ---
    r5_fairness_slider = mo.ui.slider(
        1, 200, value=150, 
        label="Tough Shift Fairness (R5) Priority:"
    )
    r6_consistency_slider = mo.ui.slider(
        1, 50, value=1, 
        label="D-Shift Consistency (R6) Priority:"
    )

    r7_consistency_slider = mo.ui.slider(
        1, 50, value=1,  # Very low default value
        label="N-Shift Consistency (R7) Priority:"
    )

    # Compose the controls stack
    controls_list = [number_of_amb_input]
    if division_selector is not None:
        controls_list.append(division_selector)
    controls_list.append(run_button)
    controls_list.append(mo.md("---"))
    controls_list.append(r5_fairness_slider)
    controls_list.append(r6_consistency_slider)
    controls_list.append(r7_consistency_slider)


    if df_input is not None and df_shifts is not None:
        solver_controls = mo.vstack(controls_list)
    else:
        solver_controls = mo.md("*Solver controls will appear after loading data*")

    mo.md("input_error") if input_error else solver_controls


    return (
        division_selector,
        get_solving,
        number_of_amb_input,
        r5_fairness_slider,
        r6_consistency_slider,
        r7_consistency_slider,
        run_button,
        set_solving,
    )


@app.cell
def _(df_input, division_selector):
    # Filtered INPUT DataFrame
    if df_input is not None and division_selector is not None and division_selector.value is not None:
        df_input_filtered = df_input[df_input["Division"] == division_selector.value].reset_index(drop=True)
    else:
        df_input_filtered = df_input

    return (df_input_filtered,)


@app.cell
def _(
    col_date,
    compute_problem_complexity,
    cp_model,
    defaultdict,
    df_input_filtered,
    df_shifts,
    get_solving,
    mo,
    number_of_amb_input,
    pd,
    r5_fairness_slider,
    r6_consistency_slider,
    r7_consistency_slider,
    run_button,
    set_solving,
    timedelta,
):
    # cell 5: Core Solver Logic (Optimized)
    # ------------------------------------------------------------------------
    # OPTIMIZATIONS APPLIED:
    # - R5 fairness uses squared deviation (faster for CP-SAT than abs deviation)
    # - Tight bounds and AddHint for fairness variables
    # - Decision strategy: assign variables in fixed order (CHOOSE_FIRST, SELECT_MIN_VALUE)
    # - Parallel search: num_search_workers=4 (uses 4 CPU cores)
    # - All other logic matches the 'stable' version
    # ------------------------------------------------------------------------

    if (run_button.value and not get_solving() and 
        df_input_filtered is not None and df_shifts is not None and 
        number_of_amb_input.value > 0):

        try:
            set_solving(True)
            mo.output.replace(mo.md("🔄 **Solving with optimized logic...**"))

            # === DATA PREPARATION ===
            _dates = sorted(df_input_filtered[col_date].unique())
            _shifts = df_shifts['Shift'].tolist() + ['O']
            _ambulances = list(range(1, number_of_amb_input.value + 1))
            _num_ambulances = len(_ambulances)
            _shift_attrs = {row['Shift']: {'type': row['Type'], 'flexible': row['Flexible'], 'delta': row['Working_Hour_Delta'], 'base_hours': row['Working_Hour']} for _, row in df_shifts.iterrows()}
            _shift_attrs['O'] = {'type': 'O', 'flexible': False, 'delta': 0, 'base_hours': 0}
            _demand = {(_row[col_date], _row['Shift']): _row['DAA'] for _, _row in df_input_filtered.iterrows()}
            _weeks = defaultdict(list)
            for _d in _dates: _weeks[_d - timedelta(days=_d.weekday())].append(_d)

            # === MODEL CREATION ===
            _model = cp_model.CpModel()
            _assign = {(_d, _a, _s): _model.NewBoolVar(f'assign_{_d}_{_a}_{_s}')
                       for _d in _dates for _a in _ambulances for _s in _shifts}
            _hour_adjust = {
                (_d, _a, _s): (
                    _model.NewIntVar(-_shift_attrs[_s]['delta'], _shift_attrs[_s]['delta'], f'adj_{_d}_{_a}_{_s}')
                    if _shift_attrs[_s]['flexible'] else _model.NewConstant(0)
                )
                for _d in _dates for _a in _ambulances for _s in _shifts
            }

            # === HARD CONSTRAINTS (R1-R4) ===
            for _d in _dates:
                for _s in _shifts:
                    if _s != 'O' and (_d, _s) in _demand:
                        _model.Add(sum(_assign[(_d, _a, _s)] for _a in _ambulances) == _demand[(_d, _s)])
            for _d in _dates:
                for _a in _ambulances:
                    _model.Add(sum(_assign[(_d, _a, _s)] for _s in _shifts) == 1)
            for _a in _ambulances:
                for _i, _d in enumerate(_dates[:-1]):
                    _next_day = _dates[_i+1]
                    for _s in [_s for _s, attrs in _shift_attrs.items() if attrs['type'] == 'N']:
                        _model.AddBoolOr([_assign[(_d, _a, _s)].Not(), _assign[(_next_day, _a, 'O')]])
            for _week_start, _week_dates in _weeks.items():
                for _a in _ambulances:
                    _weekly_hours_expr = []
                    for _d in _week_dates:
                        for _s in _shifts:
                            _actual_hours_var = _model.NewIntVar(0, 24, f'hours_{_a}_{_d}_{_s}')
                            _model.Add(_actual_hours_var == 0).OnlyEnforceIf(_assign[(_d, _a, _s)].Not())
                            _model.Add(_actual_hours_var == _shift_attrs[_s]['base_hours'] + _hour_adjust[(_d, _a, _s)]).OnlyEnforceIf(_assign[(_d, _a, _s)])
                            _weekly_hours_expr.append(_actual_hours_var)
                    _model.Add(sum(_weekly_hours_expr) == 48)

            # === SOFT CONSTRAINTS (R5 and R6) ===
            _objective_terms = []

             # === R5: Fairness of Tough Shifts (Squared Deviation, Hints, Tight Bounds) ===
        
            # Dynamically determine all shift codes of type "N" from the SHIFTS sheet
            _tough_shifts = [row['Shift'] for _, row in df_shifts.iterrows() if row['Type'] == 'N']
        
            _total_tough_shifts_to_assign = sum(_demand.get((_d, _s), 0) for _d in _dates for _s in _tough_shifts)
            min_tough = _total_tough_shifts_to_assign // _num_ambulances
            max_tough = min_tough + 1 if _total_tough_shifts_to_assign % _num_ambulances else min_tough
        
            _tough_shift_counts = [
                _model.NewIntVar(min_tough, max_tough, f'tough_count_{_a}') for _a in _ambulances
            ]
        
            for _a_idx, _a in enumerate(_ambulances):
                _model.Add(_tough_shift_counts[_a_idx] == sum(
                    _assign[(_d, _a, _s)] for _d in _dates for _s in _tough_shifts
                ))
        
            for _count_var in _tough_shift_counts:
                _model.AddHint(_count_var, min_tough)
        
            fairness_sq_devs = []
            for _count_var in _tough_shift_counts:
                _diff = _model.NewIntVar(-max_tough, max_tough, f'diff_{_count_var.Name()}')
                _model.Add(_diff == _count_var - min_tough)
                _sq_dev = _model.NewIntVar(0, max_tough * max_tough, f'sq_dev_{_count_var.Name()}')
                _model.AddMultiplicationEquality(_sq_dev, [_diff, _diff])
                fairness_sq_devs.append(_sq_dev)
        
            _objective_terms.append(r5_fairness_slider.value * sum(fairness_sq_devs))

            # === R6: Consistency of Day Shifts (Soft) ===
            if r6_consistency_slider is not None:
                for _a in _ambulances:
                    _d_shifts_for_this_amb = []
                    for _d_shift in [s for s, attrs in _shift_attrs.items() if attrs['type'] == 'D']:
                        _is_used = _model.NewBoolVar(f'used_{_a}_{_d_shift}')
                        _model.Add(sum(_assign[(_d, _a, _d_shift)] for _d in _dates) > 0).OnlyEnforceIf(_is_used)
                        _model.Add(sum(_assign[(_d, _a, _d_shift)] for _d in _dates) == 0).OnlyEnforceIf(_is_used.Not())
                        _d_shifts_for_this_amb.append(_is_used)
                    _objective_terms.append(r6_consistency_slider.value * (sum(_d_shifts_for_this_amb) - 1))

            # === Decision Strategy for Assignment Variables ===
            assign_vars = [_assign[(_d, _a, _s)] for _d in _dates for _a in _ambulances for _s in _shifts]
            _model.AddDecisionStrategy(assign_vars, cp_model.CHOOSE_FIRST, cp_model.SELECT_MIN_VALUE)

           # === SHOW PROBLEM COMPLEXITY ===
            complexity = compute_problem_complexity(_dates, _shifts, _ambulances, _shift_attrs, df_input_filtered)
            mo.output.replace(mo.vstack([
                mo.md("🔄 **Solving...**"),
                complexity,
            ]))

            # === R7: Consistency of N Shifts (Soft) ===
            # Penalize ambulances for being assigned more than one type of "N" shift
        
            if r7_consistency_slider is not None:
                # Find all shift codes of type "N"
                n_shift_types = [s for s, attrs in _shift_attrs.items() if attrs['type'] == 'N']
                for _a in _ambulances:
                    n_shifts_for_this_amb = []
                    for n_shift in n_shift_types:
                        _is_used = _model.NewBoolVar(f'usedN_{_a}_{n_shift}')
                        _model.Add(sum(_assign[(_d, _a, n_shift)] for _d in _dates) > 0).OnlyEnforceIf(_is_used)
                        _model.Add(sum(_assign[(_d, _a, n_shift)] for _d in _dates) == 0).OnlyEnforceIf(_is_used.Not())
                        n_shifts_for_this_amb.append(_is_used)
                    # Penalize using more than one type of N shift
                    _objective_terms.append(r7_consistency_slider.value * (sum(n_shifts_for_this_amb) - 1))


            # === Solve (with parallelism) ===
            if _objective_terms:
                _model.Minimize(sum(_objective_terms))
            _solver = cp_model.CpSolver()
            _solver.parameters.max_time_in_seconds = 240.0
            _solver.parameters.num_search_workers = 4  # Use 4 CPU cores for parallel search
            _status = _solver.Solve(_model)

            # === EXTRACT AND DISPLAY RESULTS ===
            if _status in (cp_model.OPTIMAL, cp_model.FEASIBLE):
                _schedule_data = []
                for _d in _dates:
                    for _a in _ambulances:
                        for _s in _shifts:
                            if _solver.Value(_assign[(_d, _a, _s)]):
                                _adjustment = _solver.Value(_hour_adjust[(_d, _a, _s)]) if _shift_attrs[_s]['flexible'] else 0
                                _actual_hours = _shift_attrs[_s]['base_hours'] + _adjustment
                                _schedule_data.append({'Date': _d, 'Ambulance': f'A{_a:03d}', 'Shift': _s, 'Base_Hours': _shift_attrs[_s]['base_hours'], 'Hour_Adjustment': _adjustment, 'Actual_Hours': _actual_hours})
                df_schedule = pd.DataFrame(_schedule_data)
                df_schedule['WeekStart'] = df_schedule['Date'].apply(lambda x: x - timedelta(days=x.weekday()))
                _weekly_totals = df_schedule.groupby(['WeekStart', 'Ambulance'])['Actual_Hours'].sum()
                _obj_val = _solver.ObjectiveValue() if _objective_terms else "N/A (No soft constraints)"
                schedule_result = mo.vstack([
                    mo.md(f"✅ **Optimal Solution Found!** (Objective Value: {_obj_val})"),
                    mo.md("### Detailed Schedule:"), mo.ui.table(df_schedule),
                    mo.md("### Weekly Hours Verification:"), mo.ui.table(_weekly_totals.reset_index()),
                ])
            else:
                schedule_result = mo.md(f"❌ **INFEASIBLE:** No solution exists that satisfies all hard constraints.")
                df_schedule = None

        finally:
            set_solving(False)

    elif not get_solving():
        schedule_result = mo.md("*Click 'Generate Schedule' to solve*")
        df_schedule = None
        schedule_result
    else:
        df_schedule = None
        schedule_result = None

    schedule_result

    return (df_schedule,)


@app.cell
def _(date, df_schedule, mo, pd, timedelta):
    # cell 6: Calendar View
    if df_schedule is not None and not df_schedule.empty:
        _calendar_sections = []
        _tough_shifts = ['H', 'N']
        for _week_start, _week_group in df_schedule.groupby('WeekStart'):
            _week_group_copy = _week_group.copy()
            _week_group_copy['display_shift'] = _week_group_copy.apply(lambda row: f"{row['Shift']} ({row['Actual_Hours']}h)", axis=1)
            _calendar_df = _week_group_copy.pivot_table(index='Ambulance', columns='Date', values='display_shift', aggfunc='first', fill_value="")
            _total_hours = _week_group.groupby('Ambulance')['Actual_Hours'].sum()
            _calendar_df['Weekly Total Hours'] = _total_hours
            _tough_shift_counts = _week_group[_week_group['Shift'].isin(_tough_shifts)].groupby('Ambulance').size()
            _calendar_df['Total H+N Shifts'] = _tough_shift_counts.fillna(0).astype(int)
            _calendar_df = _calendar_df.sort_index()
            _formatted_columns = {_col: _col.strftime('%a %m/%d') for _col in _calendar_df.columns if isinstance(_col, (pd.Timestamp, date))}
            _calendar_df = _calendar_df.rename(columns=_formatted_columns)
            _week_end = _week_start + timedelta(days=6)
            _week_header = f"### Week: {_week_start.strftime('%Y-%m-%d')} to {_week_end.strftime('%Y-%m-%d')}"
            _calendar_sections.extend([mo.md(_week_header), mo.ui.table(_calendar_df), mo.md("---")])
        calendar_view = mo.vstack(_calendar_sections)
    else:
        calendar_view = mo.md("*Calendar view will appear after generating schedule*")
    calendar_view

    return


@app.cell
def _(df_input_filtered, df_schedule, mo, pd):
    # cell 7: DAA Requirement Verification
    if df_input_filtered is not None and df_schedule is not None and not df_schedule.empty:
        _scheduled_counts = df_schedule.groupby(['Date', 'Shift']).size().reset_index(name='Scheduled')
        _daa_summary_df = pd.merge(df_input_filtered, _scheduled_counts, on=['Date', 'Shift'], how='left')
        _daa_summary_df['Scheduled'] = _daa_summary_df['Scheduled'].fillna(0).astype(int)
        def _determine_status(row):
            if row['Scheduled'] == row['DAA']: return '✅ Exactly Met'
            elif row['Scheduled'] > row['DAA']: return '🟡 Exceeded'
            else: return '🔴 Missed'
        _daa_summary_df['Status'] = _daa_summary_df.apply(_determine_status, axis=1)
        _final_summary = _daa_summary_df[['Date', 'Shift', 'DAA', 'Scheduled', 'Status']].rename(columns={'DAA': 'Required'})
        daa_check_view = mo.vstack([
            mo.md("### DAA Requirement Verification"),
            mo.ui.table(_final_summary)
        ])
    else:
        daa_check_view = mo.md("*DAA check will appear after generating schedule*")
    daa_check_view
    return


@app.cell
def _(mo):
    export_format = mo.ui.dropdown(
        options=["XLSX", "PDF"],
        value="XLSX",
        label="Export calendar as:"
    )
    export_run_button = mo.ui.run_button(label="Export Calendar")
    return export_format, export_run_button


@app.cell
def _(
    date,
    df_schedule,
    export_format,
    export_run_button,
    io,
    mo,
    pd,
    timedelta,
):
    if df_schedule is not None and not df_schedule.empty:
        export_status = mo.md("")
        export_data = None

        # Prepare weekly calendar tables
        week_tables = {}
        _tough_shifts = ['H', 'N']
        for _week_start, _week_group in df_schedule.groupby('WeekStart'):
            _week_group_copy = _week_group.copy()
            _week_group_copy['display_shift'] = _week_group_copy.apply(
                lambda row: f"{row['Shift']} ({row['Actual_Hours']}h)", axis=1
            )
            _calendar_df = _week_group_copy.pivot_table(
                index='Ambulance', columns='Date', values='display_shift', aggfunc='first', fill_value=""
            )
            _total_hours = _week_group.groupby('Ambulance')['Actual_Hours'].sum()
            _calendar_df['Weekly Total Hours'] = _total_hours
            _tough_shift_counts = _week_group[_week_group['Shift'].isin(_tough_shifts)].groupby('Ambulance').size()
            # Ensure all ambulances are present and NaN is replaced by 0
            _calendar_df['Total H+N Shifts'] = _tough_shift_counts.reindex(_calendar_df.index).fillna(0).astype(int)
            _calendar_df = _calendar_df.sort_index()
            _formatted_columns = {_col: _col.strftime('%a %m/%d') for _col in _calendar_df.columns if isinstance(_col, (pd.Timestamp, date))}
            _calendar_df = _calendar_df.rename(columns=_formatted_columns)
            week_tables[_week_start] = _calendar_df

        if export_run_button.value:
            if export_format.value == "XLSX":
                buf = io.BytesIO()
                with pd.ExcelWriter(buf, engine='xlsxwriter') as writer:
                    for week_start, week_df in week_tables.items():
                        week_end = week_start + timedelta(days=6)
                        sheet_name = f"{week_start.strftime('%Y-%m-%d')}_to_{week_end.strftime('%m-%d')}"
                        # Excel sheet names max length is 31
                        writer.book.add_worksheet(sheet_name[:31])
                        week_df.to_excel(writer, sheet_name=sheet_name[:31])
                buf.seek(0)
                export_data = mo.download(
                    buf.getvalue(),
                    filename="calendar.xlsx",
                    label="Download XLSX"
                )
                export_status = mo.md("✅ XLSX calendar ready for download (one week per sheet).")
            elif export_format.value == "PDF":
                try:
                    import matplotlib.pyplot as plt
                    from matplotlib.backends.backend_pdf import PdfPages

                    buf = io.BytesIO()
                    with PdfPages(buf) as pdf:
                        for week_start, week_df in week_tables.items():
                            week_end = week_start + timedelta(days=6)
                            fig, ax = plt.subplots(figsize=(min(20, 2 + len(week_df.columns)), min(20, 2 + len(week_df))))
                            ax.axis('tight')
                            ax.axis('off')
                            table = ax.table(
                                cellText=week_df.values,
                                colLabels=week_df.columns,
                                rowLabels=week_df.index,
                                loc='center'
                            )
                            table.auto_set_font_size(False)
                            table.set_fontsize(8)
                            plt.title(f"Week: {week_start.strftime('%Y-%m-%d')} to {week_end.strftime('%Y-%m-%d')}")
                            pdf.savefig(fig, bbox_inches='tight')
                            plt.close(fig)
                    buf.seek(0)
                    export_data = mo.download(
                        buf.getvalue(),
                        filename="calendar.pdf",
                        label="Download PDF"
                    )
                    export_status = mo.md("✅ PDF calendar ready for download (one week per page).")
                except ImportError:
                    export_status = mo.md("❌ PDF export requires matplotlib. Please install it in your environment.")

        export_controls = mo.vstack([
            mo.hstack([export_format, export_run_button]),
            export_status,
            export_data if export_data else mo.md("")
        ])
    else:
        export_controls = mo.md("*No schedule to export yet.*")

    export_controls

    return


@app.cell
def _(mo):
    # cell 9: Complexity Metric Function

    def compute_problem_complexity(_dates, _shifts, _ambulances, _shift_attrs, df_input):
        # Assignment variables
        num_assign_vars = len(_dates) * len(_ambulances) * len(_shifts)

        # Hard constraints
        num_hard_demand_constraints = len(_dates) * (len(_shifts) - 1)
        num_hard_one_shift_constraints = len(_dates) * len(_ambulances)
        num_night_shifts = sum(1 for s in _shifts if _shift_attrs[s]['type'] == 'N')
        num_hard_night_rest_constraints = len(_ambulances) * (len(_dates) - 1) * num_night_shifts
        num_weeks = len(_dates) // 7
        num_hard_weekly_hours_constraints = len(_ambulances) * num_weeks
        num_hard_constraints = (
            num_hard_demand_constraints +
            num_hard_one_shift_constraints +
            num_hard_night_rest_constraints +
            num_hard_weekly_hours_constraints
        )

        # Soft constraints
        num_soft_fairness_constraints = len(_ambulances) * 3
        num_day_shift_types = sum(1 for s in _shifts if _shift_attrs[s]['type'] == 'D')
        num_soft_consistency_constraints = len(_ambulances) * num_day_shift_types
        num_soft_constraints = num_soft_fairness_constraints + num_soft_consistency_constraints

        return mo.md(f"""
        **Problem Size:**  
        - **Assignment Variables:** {num_assign_vars:,}  
        - **Ambulances:** {len(_ambulances)}  
        - **Days:** {len(_dates)}  
        - **Shifts:** {len(_shifts)}

        **Constraints:**  
        - **Hard Constraints:** {num_hard_constraints:,}  
        - **Soft Constraints:** {num_soft_constraints:,}
        """)

    return (compute_problem_complexity,)


if __name__ == "__main__":
    app.run()
