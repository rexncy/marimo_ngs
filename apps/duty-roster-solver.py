import marimo

__generated_with = "0.15.2"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
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


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
    /// details | Ambulance Roster Scheduling Solver

    # **Solver Objective**
        To generate a fair and optimal weekly work roster for a fleet of ambulances that strictly adheres to all operational demands and work rules, while strongly optimizing for fairness and shift consistency.

    ## **Hard Requirements**

        *   **H1: Exactly Meet Demand**
            The number of scheduled ambulances for each shift (`H`, `N`, `Dv`, etc.) must be *exactly equal* to the DAA specified in the `INPUT` sheet.

        *   **H2: One Shift Per Day**
            Each ambulance must be assigned exactly one shift per day (this includes rest shifts like 'O').

        *   **H3: Night Shift Rest**
            An ambulance that works a night shift (`Type` 'N') *must* be assigned an "Off" shift ('O') on the immediately following day.

        *   **H4: 48-Hour Work Week**
            Each ambulance must work either (i) **exactly 48 hours** or (ii) an **average of 48 hours** per week (Monday to Sunday). The solver is allowed to use the flexibility of `Flexible` shifts (adjusting their hours within the `Working_Hour_Delta`) to achieve this target.

    ## **Soft Goals**

        *   **S1: Fairness of Tough Shifts**
            The total number of "tough" shifts ('H' and 'N') should be distributed as evenly as possible among all ambulances. The solver will strongly penalize imbalance, but perfect equality is not strictly required.

        *   **S2: Consistency of Day Shifts**
            The solver should try to assign as few different types of "D" shifts as possible to the same ambulance. The importance of this goal is controlled by the "D-Shift Consistency Priority" slider in the user interface.

        *   **S3: Consistency of Night Shifts**
            The solver should try to assign as few different types of "N" shifts as possible to the same ambulance. The importance of this goal is controlled by the "N-Shift Consistency Priority" slider in the user interface.

    ///
    """
    )
    return


@app.cell(hide_code=True)
def _(Path, mo):
    # Global UI elements for cross-cell reference
    FILE_DIR = "apps/files/duty-roster-solver-files"

    file_browser = mo.ui.file_browser(
        initial_path=Path(FILE_DIR),
        filetypes=[".xlsx"],
        restrict_navigation=True,
        multiple=False,
        label="Select an existing roster file:",
    )

    file_uploader = mo.ui.file(
        filetypes=[".xlsx"], label="Or upload your own roster file (.xlsx):"
    )


    file_input_tabs = mo.ui.tabs(
        {
            "Choose Existing": file_browser,
            "Upload New": file_uploader,
        }
    )

    file_input_tabs
    return FILE_DIR, file_browser, file_uploader


@app.cell
def _(file_browser, mo):
    delete_button = mo.ui.run_button(label="Delete selected file")

    _download = mo.download(
        data=lambda: open(file_browser.path(index=0), "rb"),
        filename=file_browser.path(index=0).name if file_browser.value else '',
        mimetype="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        label="Download Selected file"
    )

    h_stack = mo.hstack(items=[delete_button, _download],align="start")

    h_stack if file_browser.value else None
    return (delete_button,)


@app.cell
def _(delete_button, file_browser, mo):
    if delete_button.value and file_browser.path():
        import os
        _selected_path = str(file_browser.path())
        if os.path.isfile(_selected_path):
            os.remove(_selected_path)
            mo.output.replace(f"Deleted: {_selected_path}")
    return


@app.cell(hide_code=True)
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
        _file_status = mo.md(
            f"🟡 Using selected file: **{file_browser.path().name}**"
        )
    else:
        _file_status = mo.md(
            "⚠️ **Please select an existing file or upload your own roster file (.xlsx) to proceed!**"
        )

    if _excel_file is not None:
        # Read and process data
        df_input = pd.read_excel(_excel_file, sheet_name="INPUT")
        col_date = df_input.columns[0]
        df_input[col_date] = pd.to_datetime(df_input[col_date]).dt.date

        df_shifts = pd.read_excel(_excel_file, sheet_name="SHIFTS")

        _tabs = mo.ui.tabs(
            {
                "INPUT Sheet": mo.ui.table(df_input),
                "SHIFTS Sheet": mo.ui.table(df_shifts),
            }
        )
        # Display status and data
        data_display = mo.vstack([_file_status, _tabs])
    else:
        data_display = _file_status

    data_display
    return col_date, df_input, df_shifts


@app.cell(hide_code=True)
def _(df_input, df_shifts, mo):
    input_error = False
    if df_input is not None and df_shifts is not None:
        input_shifts = set(df_input["Shift"].unique())
        shifts_shifts = set(df_shifts["Shift"].unique())
        invalid_shifts = input_shifts - shifts_shifts

        if invalid_shifts:
            input_error = True
            validation_msg = mo.md(
                f"❌ **Validation Error:** The following shifts in INPUT are not defined in the SHIFTS sheet: "
                f"`{', '.join(sorted(invalid_shifts))}`"
            )
        else:
            validation_msg = mo.md(
                "✅ All shift codes in INPUT are valid and defined in the SHIFTS sheet."
            )
    else:
        validation_msg = mo.md(
            "*Load both INPUT and SHIFTS sheets to validate shift codes.*"
        )

    validation_msg
    return (input_error,)


@app.cell(hide_code=True)
def _(df_input, mo):
    # cell 4: Solver Settings
    # Define the state variable to track if the solver is running.
    get_solving, set_solving = mo.state(False)

    timeout_slider = mo.ui.slider(
        30, 180, value=90, label="Max run time in seconds:"
    )

    number_of_amb_input = mo.ui.number(
        start=1,
        stop=100,
        step=1,
        value=48,  # Default: 48 ambulances
        label="Total ambulances available:",
    )

    # Division selector (only if present)
    if df_input is not None and "Division" in df_input.columns:
        division_values = sorted(df_input["Division"].dropna().unique())
        division_selector = mo.ui.dropdown(
            options=division_values,
            value=division_values[0] if division_values else None,
            label="Select Division:",
        )
    else:
        division_selector = None

    allow_double_nights_checkbox = mo.ui.checkbox(
        value=True,
        label="Allow double nights (permit two consecutive 'N' shifts, but not three)",
    )

    allow_triple_nights_checkbox = mo.ui.checkbox(
        value=False,
        label="Allow triple nights (permit three consecutive 'N' shifts, but not four)",
    )

    r4_average_checkbox = mo.ui.checkbox(
        value=True,
        label="Allow average 48 hours per week (over the whole period)",
    )

    r4_leq_checkbox = mo.ui.checkbox(
        value=True,
        label="Allow working hours to be ≤ 48 (instead of exactly 48)",
    )

    # The run_button is disabled when get_solving() is True.
    run_button = mo.ui.run_button(
        label="🚀 Generate Schedule", disabled=get_solving()
    )

    S1_fairness_checkbox = mo.ui.checkbox(
        value=True,
        label="Enable Tough Shift Fairness (S1) Priority:",
    )
    # --- Priority Sliders for Soft Constraints ---
    S1_fairness_slider = mo.ui.slider(
        1, 200, value=150, label="Tough Shift Fairness (S1) Priority"
    )

    S2_consistency_checkbox = mo.ui.checkbox(
        value=True,
        label="Enable D-Shift Consistency (S2) Priority",
    )
    S2_consistency_slider = mo.ui.slider(
        1, 50, value=35, label="D-Shift Consistency (S2) Priority:"
    )

    S3_consistency_checkbox = mo.ui.checkbox(
        value=True,
        label="Enable N-Shift Consistency (S3) Priority",
    )
    S3_consistency_slider = mo.ui.slider(
        1,
        50,
        value=10,  # Very low default value
        label="N-Shift Consistency (S3) Priority:",
    )
    S4_balance_slider = mo.ui.slider(
        1,
        200,
        value=150,  # Very low default value
        label="Hour under Max Fairness (S4) Priority:",
    )
    return (
        S1_fairness_checkbox,
        S1_fairness_slider,
        S2_consistency_checkbox,
        S2_consistency_slider,
        S3_consistency_checkbox,
        S3_consistency_slider,
        S4_balance_slider,
        allow_double_nights_checkbox,
        allow_triple_nights_checkbox,
        division_selector,
        get_solving,
        number_of_amb_input,
        r4_average_checkbox,
        r4_leq_checkbox,
        run_button,
        set_solving,
        timeout_slider,
    )


@app.cell(hide_code=True)
def _(
    S1_fairness_checkbox,
    S1_fairness_slider,
    S2_consistency_checkbox,
    S2_consistency_slider,
    S3_consistency_checkbox,
    S3_consistency_slider,
    S4_balance_slider,
    allow_double_nights_checkbox,
    allow_triple_nights_checkbox,
    df_input,
    df_shifts,
    division_selector,
    input_error,
    mo,
    number_of_amb_input,
    r4_average_checkbox,
    r4_leq_checkbox,
    run_button,
    timeout_slider,
):
    # Compose the controls stack
    controls_list = [timeout_slider]
    controls_list.append(number_of_amb_input)
    if division_selector is not None:
        controls_list.append(division_selector)
    controls_list.append(run_button)
    controls_list.append(mo.md("---"))
    controls_list.append(r4_average_checkbox)
    controls_list.append(r4_leq_checkbox)
    controls_list.append(allow_double_nights_checkbox)
    controls_list.append(allow_triple_nights_checkbox)
    controls_list.append(S1_fairness_checkbox)
    if S1_fairness_checkbox.value:
        controls_list.append(S1_fairness_slider)
    controls_list.append(S2_consistency_checkbox)
    if S2_consistency_checkbox.value:
        controls_list.append(S2_consistency_slider)
    controls_list.append(S3_consistency_checkbox)
    if S3_consistency_checkbox.value:
        controls_list.append(S3_consistency_slider)
    if r4_leq_checkbox.value:
        controls_list.append(S4_balance_slider)


    if df_input is not None and df_shifts is not None:
        solver_controls = mo.vstack(controls_list)
    else:
        solver_controls = mo.md("*Solver controls will appear after loading data*")

    mo.md("input_error") if input_error else solver_controls
    return


@app.cell(hide_code=True)
def _(df_input, division_selector):
    # Filtered INPUT DataFrame
    if (
        df_input is not None
        and division_selector is not None
        and division_selector.value is not None
    ):
        df_input_filtered = df_input[
            df_input["Division"] == division_selector.value
        ].reset_index(drop=True)
    else:
        df_input_filtered = df_input
    return (df_input_filtered,)


@app.cell
def _():
    HOUR_SCALE = 2  # half-hour
    def scale(x):
        return int(round(x * HOUR_SCALE))
    def unscale(x):
        return x / HOUR_SCALE
    return scale, unscale


@app.cell
def _(
    S1_fairness_checkbox,
    S1_fairness_slider,
    S2_consistency_checkbox,
    S2_consistency_slider,
    S3_consistency_checkbox,
    S3_consistency_slider,
    S4_balance_slider,
    allow_double_nights_checkbox,
    allow_triple_nights_checkbox,
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
    r4_average_checkbox,
    r4_leq_checkbox,
    run_button,
    scale,
    set_solving,
    timedelta,
    timeout_slider,
    unscale,
):
    # cell 5: Core Solver Logic (Optimized)
    # ------------------------------------------------------------------------
    # OPTIMIZATIONS APPLIED:
    # - S1 fairness uses squared deviation (faster for CP-SAT than abs deviation)
    # - Tight bounds and AddHint for fairness variables
    # - Decision strategy: assign variables in fixed order (CHOOSE_FIRST, SELECT_MIN_VALUE)
    # - Parallel search: num_search_workers=4 (uses 4 CPU cores)
    # - Configurable double nights (R3) with correct "O" after double "N"
    # - S1, S2, S3 soft constraints
    # - Problem complexity display and infeasibility reason reporting
    # ------------------------------------------------------------------------

    if (
        run_button.value
        and not get_solving()
        and df_input_filtered is not None
        and df_shifts is not None
        and number_of_amb_input.value > 0
    ):

        try:
            set_solving(True)
            mo.output.replace(mo.md("🔄 **Solving with optimized logic...**"))

            # === DATA PREPARATION ===
            _dates = sorted(df_input_filtered[col_date].unique())
            _shifts = df_shifts["Shift"].tolist() + ["O"]
            _ambulances = list(range(1, number_of_amb_input.value + 1))
            _num_ambulances = len(_ambulances)
            # All deltas and bases must be scaled into the model.
            _shift_attrs = {
                row["Shift"]: {
                    "type": row["Type"],
                    "flexible": row["Flexible"],
                    "delta": scale(row["Working_Hour_Delta"]),
                    "base_hours": scale(row["Working_Hour"]),
                }
                for _, row in df_shifts.iterrows()
            }

            _shift_attrs["O"] = {
                "type": "O",
                "flexible": False,
                "delta": 0,
                "base_hours": 0,
            }
            _demand = {
                (_row[col_date], _row["Shift"]): _row["DAA"]
                for _, _row in df_input_filtered.iterrows()
            }
            # week starts on monday:
            # typical calendar day
            """
            _weeks = defaultdict(list)
            for _d in _dates:
                _weeks[_d - timedelta(days=_d.weekday())].append(_d)
            """

            # week starts on the earliest date in _dates
            # rolling 7-day fixed window schedule
            _weeks = defaultdict(list)
            if _dates:
                _start_date = _dates[0]
                for _d in _dates:
                    # Compute which week this date belongs to
                    _week_index = (_d - _start_date).days // 7
                    _week_start = _start_date + timedelta(days=_week_index * 7)
                    _weeks[_week_start].append(_d)

           # NEW: Validation for exactly one pair per day when pair exists
            _h_names = [row["Shift"] for _, row in df_shifts.iterrows() if row["Shift"].startswith("H")]
            _n_names = [row["Shift"] for _, row in df_shifts.iterrows() if row["Shift"].startswith("N")]

            def _suffix(name):
                return name[1:] if len(name) >= 2 else ""

            _h_by_code = { _suffix(h): h for h in _h_names }
            _n_by_code = { _suffix(n): n for n in _n_names }
            _hn_pairs = { code: (_h_by_code[code], _n_by_code[code]) for code in set(_h_by_code).intersection(_n_by_code) }

            invalid_inputs = []
            for _d in _dates:
                for _code, (_h, _n) in _hn_pairs.items():
                    dem_h = _demand.get((_d, _h), 0)
                    dem_n = _demand.get((_d, _n), 0)
                    if dem_h > 0 and dem_n > 0:
                        if dem_h != 1 or dem_n != 1:
                            invalid_inputs.append(f"Day {_d}: Pair for code '{_code}' must be exactly one (dem_h={dem_h}, dem_n={dem_n}) - multiples or unequals not allowed.")
            if invalid_inputs:
                mo.output.replace(mo.md("❌ **Invalid Input:**\n" + "\n".join(invalid_inputs)))
                set_solving(False)
                raise ValueError("Invalid hotel pair demands - solving aborted.")


            # === MODEL CREATION ===
            _model = cp_model.CpModel()
            _assign = {
                (_d, _a, _s): _model.NewBoolVar(f"assign_{_d}_{_a}_{_s}")
                for _d in _dates
                for _a in _ambulances
                for _s in _shifts
            }
            # no need scale() the range is already using the scaled delta from above
            _hour_adjust = {
                (_d, _a, _s): (
                    _model.NewIntVar(
                        -_shift_attrs[_s]["delta"],
                        _shift_attrs[_s]["delta"],
                        f"adj_{_d}_{_a}_{_s}",
                    )
                    if _shift_attrs[_s]["flexible"]
                    else _model.NewConstant(0)
                )
                for _d in _dates
                for _a in _ambulances
                for _s in _shifts
            }

            # === HARD CONSTRAINTS (H1-H4) ===
            for _d in _dates:
                for _s in _shifts:
                    if _s != "O" and (_d, _s) in _demand:
                        _model.Add(
                            sum(_assign[(_d, _a, _s)] for _a in _ambulances)
                            == _demand[(_d, _s)]
                        )
            for _d in _dates:
                for _s in _shifts:
                    if _s != "O" and ((_d, _s) not in _demand or _demand[(_d, _s)] == 0):
                        for _a in _ambulances:
                            if (_d, _a, _s) in _assign:
                                _model.Add(_assign[(_d, _a, _s)] == 0)


            for _d in _dates:
                for _a in _ambulances:
                    _model.Add(
                        sum(_assign[(_d, _a, _s)] for _s in _shifts) == 1
                    )

            # === H3: Night Shift Rest (Configurable Double Nights, Enforce O after double N) ===
            n_shift_types = [s for s, attrs in _shift_attrs.items() if attrs["type"] == "N"]

            if allow_triple_nights_checkbox.value:
                for _a in _ambulances:
                    for _i in range(len(_dates) - 3):
                        d0, d1, d2, d3 = _dates[_i], _dates[_i + 1], _dates[_i + 2], _dates[_i + 3]
                        for n_shift in n_shift_types:
                            # Forbid four consecutive N
                            if all((_d, _a, n_shift) in _assign for _d in [d0, d1, d2, d3]):
                                _model.Add(
                                    sum(_assign[(_d, _a, n_shift)] for _d in [d0, d1, d2, d3]) <= 3
                                )
                            # If N on d0, d1, d2, then O on d3
                            if all((_d, _a, n_shift) in _assign for _d in [d0, d1, d2]) and (d3, _a, "O") in _assign:
                                _model.AddBoolOr([
                                    _assign[(d0, _a, n_shift)].Not(),
                                    _assign[(d1, _a, n_shift)].Not(),
                                    _assign[(d2, _a, n_shift)].Not(),
                                    _assign[(d3, _a, "O")]
                                ])
                    # For single or double N not followed by N, enforce O
                    for _i in range(len(_dates) - 1):
                        d0, d1 = _dates[_i], _dates[_i + 1]
                        for n_shift in n_shift_types:
                            if (d0, _a, n_shift) in _assign and (d1, _a, "O") in _assign and (d1, _a, n_shift) in _assign:
                                _model.AddBoolOr([
                                    _assign[(d0, _a, n_shift)].Not(),
                                    _assign[(d1, _a, "O")],
                                    _assign[(d1, _a, n_shift)]
                                ])
                    for _i in range(len(_dates) - 2):
                        d0, d1, d2 = _dates[_i], _dates[_i + 1], _dates[_i + 2]
                        for n_shift in n_shift_types:
                            if all((_d, _a, n_shift) in _assign for _d in [d0, d1]) and (d2, _a, "O") in _assign and (d2, _a, n_shift) in _assign:
                                _model.AddBoolOr([
                                    _assign[(d0, _a, n_shift)].Not(),
                                    _assign[(d1, _a, n_shift)].Not(),
                                    _assign[(d2, _a, "O")],
                                    _assign[(d2, _a, n_shift)]
                                ])

            elif allow_double_nights_checkbox.value:
                for _a in _ambulances:
                    for _i in range(len(_dates) - 2):
                        d0, d1, d2 = _dates[_i], _dates[_i + 1], _dates[_i + 2]
                        for n_shift in n_shift_types:
                            # Forbid three consecutive N
                            if all((_d, _a, n_shift) in _assign for _d in [d0, d1, d2]):
                                _model.Add(
                                    sum(_assign[(_d, _a, n_shift)] for _d in [d0, d1, d2]) <= 2
                                )
                            # If N on d0 and d1, then O on d2
                            if (d0, _a, n_shift) in _assign and (d1, _a, n_shift) in _assign and (d2, _a, "O") in _assign:
                                _model.AddBoolOr([
                                    _assign[(d0, _a, n_shift)].Not(),
                                    _assign[(d1, _a, n_shift)].Not(),
                                    _assign[(d2, _a, "O")]
                                ])
                    # For single N not followed by N, enforce O
                    for _i in range(len(_dates) - 1):
                        d0, d1 = _dates[_i], _dates[_i + 1]
                        for n_shift in n_shift_types:
                            if (d0, _a, n_shift) in _assign and (d1, _a, "O") in _assign and (d1, _a, n_shift) in _assign:
                                _model.AddBoolOr([
                                    _assign[(d0, _a, n_shift)].Not(),
                                    _assign[(d1, _a, "O")],
                                    _assign[(d1, _a, n_shift)]
                                ])

            else:
                for _a in _ambulances:
                    for _i in range(len(_dates) - 1):
                        d0, d1 = _dates[_i], _dates[_i + 1]
                        for n_shift in n_shift_types:
                            if (d0, _a, n_shift) in _assign and (d1, _a, "O") in _assign:
                                # If N on d0, then O on d1
                                _model.AddBoolOr([
                                    _assign[(d0, _a, n_shift)].Not(),
                                    _assign[(d1, _a, "O")]
                                ])

            # === H4 ===#
            # Prepare actual_hours_vars for all valid (_d, _a, _s)
            _actual_hours_vars = {}
            for _d in _dates:
                for _a in _ambulances:
                    for _s in _shifts:
                        if (_d, _a, _s) in _assign:
                            # scale() needed
                            _actual_hours_var = _model.NewIntVar(0, scale(24), f"hours_{_a}_{_d}_{_s}")

                            _model.Add(_actual_hours_var == 0).OnlyEnforceIf(_assign[(_d, _a, _s)].Not())
                            _model.Add(
                                _actual_hours_var == _shift_attrs[_s]["base_hours"] + _hour_adjust[(_d, _a, _s)]
                            ).OnlyEnforceIf(_assign[(_d, _a, _s)])
                            _actual_hours_vars[(_d, _a, _s)] = _actual_hours_var

            if r4_average_checkbox.value:
                # Average mode: total hours over the whole period must be ≤ 48 * number of weeks if relaxed, else ==
                num_weeks = len(_weeks)
                for _a in _ambulances:
                    _total_hours_expr = [
                        _actual_hours_vars[(_d, _a, _s)]
                        for _d in _dates for _s in _shifts
                        if (_d, _a, _s) in _actual_hours_vars
                    ]
                    # scale() the constraint bounds
                    if r4_leq_checkbox.value:
                        _model.Add(sum(_total_hours_expr) <= scale(48) * num_weeks)
                    else:
                        _model.Add(sum(_total_hours_expr) == scale(48) * num_weeks)
            else:
                # Per-week mode: each week must be ≤ 48 if relaxed, else ==
                for _week_start, _week_dates in _weeks.items():
                    for _a in _ambulances:
                        _weekly_hours_expr = [
                            _actual_hours_vars[(_d, _a, _s)]
                            for _d in _week_dates for _s in _shifts
                            if (_d, _a, _s) in _actual_hours_vars
                        ]
                        # scale() the constraint bounds
                        if r4_leq_checkbox.value:
                            _model.Add(sum(_weekly_hours_expr) <= scale(48))
                        else:
                            _model.Add(sum(_weekly_hours_expr) == scale(48))

            # === H5: Hotel shift pair add up to 24h

            for _code, (_h, _n) in _hn_pairs.items():
                if _h not in _shift_attrs or _n not in _shift_attrs:
                    continue
                for _d in _dates:
                    dem_h = _demand.get((_d, _h), 0)
                    dem_n = _demand.get((_d, _n), 0)
                    if dem_h > 0 and dem_n > 0:  # Pair exists (validated as exactly one)
                        # Sum actual hours over all ambulances for _h and _n (only assigned will be non-zero)
                        sum_h = sum(_actual_hours_vars[(_d, _a, _h)] for _a in _ambulances if (_d, _a, _h) in _actual_hours_vars)
                        sum_n = sum(_actual_hours_vars[(_d, _a, _n)] for _a in _ambulances if (_d, _a, _n) in _actual_hours_vars)
                        _model.Add(sum_h + sum_n == scale(24))



            # === SOFT CONSTRAINTS (S1, S2, S3, S4) ===
            _objective_terms = []

            # S1: Fairness of Tough Shifts (Squared Deviation, Hints, Tight Bounds)
            if S1_fairness_checkbox.value:
                _tough_shifts = [
                    row["Shift"]
                    for _, row in df_shifts.iterrows()
                    if row["Type"] in ["H", "N"]  # Revised to include both H and N
                ]
                _total_tough_shifts_to_assign = sum(
                    _demand.get((_d, _s), 0) for _d in _dates for _s in _tough_shifts
                )
                min_tough = _total_tough_shifts_to_assign // _num_ambulances
                max_tough = (
                    min_tough + 1
                    if _total_tough_shifts_to_assign % _num_ambulances
                    else min_tough
                )
                _tough_shift_counts = [
                    _model.NewIntVar(min_tough, max_tough, f"tough_count_{_a}")
                    for _a in _ambulances
                ]
                for _a_idx, _a in enumerate(_ambulances):
                    _model.Add(
                        _tough_shift_counts[_a_idx]
                        == sum(
                            _assign[(_d, _a, _s)]
                            for _d in _dates
                            for _s in _tough_shifts
                        )
                    )
                for _count_var in _tough_shift_counts:
                    _model.AddHint(_count_var, min_tough)
                fairness_sq_devs = []
                for _count_var in _tough_shift_counts:
                    _diff = _model.NewIntVar(
                        -max_tough, max_tough, f"diff_{_count_var.Name()}"
                    )
                    _model.Add(_diff == _count_var - min_tough)
                    _sq_dev = _model.NewIntVar(
                        0, max_tough * max_tough, f"sq_dev_{_count_var.Name()}"
                    )
                    _model.AddMultiplicationEquality(_sq_dev, [_diff, _diff])
                    fairness_sq_devs.append(_sq_dev)
                _objective_terms.append(
                    S1_fairness_slider.value * sum(fairness_sq_devs)
                )


            # S2: Consistency of Day Shifts (Soft)

            if S2_consistency_checkbox and S2_consistency_slider is not None:
                for _a in _ambulances:
                    _d_shifts_for_this_amb = []
                    for _d_shift in [
                        s
                        for s, attrs in _shift_attrs.items()
                        if attrs["type"] == "D"
                    ]:
                        _is_used = _model.NewBoolVar(f"used_{_a}_{_d_shift}")
                        _model.Add(
                            sum(_assign[(_d, _a, _d_shift)] for _d in _dates) > 0
                        ).OnlyEnforceIf(_is_used)
                        _model.Add(
                            sum(_assign[(_d, _a, _d_shift)] for _d in _dates) == 0
                        ).OnlyEnforceIf(_is_used.Not())
                        _d_shifts_for_this_amb.append(_is_used)
                    _objective_terms.append(
                        S2_consistency_slider.value
                        * (sum(_d_shifts_for_this_amb) - 1)
                    )

            # S3: Consistency of N Shifts (Soft)
            if S3_consistency_checkbox and S3_consistency_slider is not None:
                n_shift_types = [
                    s for s, attrs in _shift_attrs.items() if attrs["type"] == "N"
                ]
                for _a in _ambulances:
                    n_shifts_for_this_amb = []
                    for n_shift in n_shift_types:
                        _is_used = _model.NewBoolVar(f"usedN_{_a}_{n_shift}")
                        _model.Add(
                            sum(_assign[(_d, _a, n_shift)] for _d in _dates) > 0
                        ).OnlyEnforceIf(_is_used)
                        _model.Add(
                            sum(_assign[(_d, _a, n_shift)] for _d in _dates) == 0
                        ).OnlyEnforceIf(_is_used.Not())
                        n_shifts_for_this_amb.append(_is_used)
                    _objective_terms.append(
                        S3_consistency_slider.value
                        * (sum(n_shifts_for_this_amb) - 1)
                    )

            # === S4
            if S4_balance_slider is not None:
                num_weeks = len(_weeks)
                max_total_hours = scale(48) * num_weeks

                # For each ambulance, create IntVars for total hours and hours under max
                total_hours_vars = []
                hours_under_max_vars = []
                for _a in _ambulances:
                    total_hours_var = _model.NewIntVar(0, max_total_hours, f"total_hours_{_a}")
                    _model.Add(
                        total_hours_var == sum(
                            _actual_hours_vars[(_d, _a, _s)]
                            for _d in _dates for _s in _shifts
                            if (_d, _a, _s) in _actual_hours_vars
                        )
                    )
                    total_hours_vars.append(total_hours_var)
                    hours_under_max_var = _model.NewIntVar(0, max_total_hours, f"hours_under_max_{_a}")
                    _model.Add(hours_under_max_var == max_total_hours - total_hours_var)
                    hours_under_max_vars.append(hours_under_max_var)

                total_under_max = _model.NewIntVar(0, max_total_hours * _num_ambulances, "total_under_max")
                _model.Add(total_under_max == sum(hours_under_max_vars))

                mean_hours_under_max = _model.NewIntVar(0, max_total_hours, "mean_hours_under_max")
                _model.AddDivisionEquality(mean_hours_under_max, total_under_max, _num_ambulances)


                S4_sq_devs = []
                for var in hours_under_max_vars:
                    diff = _model.NewIntVar(-max_total_hours, max_total_hours, f"diff_{var.Name()}")
                    _model.Add(diff == var - mean_hours_under_max)
                    sq_dev = _model.NewIntVar(0, max_total_hours * max_total_hours, f"sq_dev_{var.Name()}")
                    _model.AddMultiplicationEquality(sq_dev, [diff, diff])
                    S4_sq_devs.append(sq_dev)

                _objective_terms.append(
                    S4_balance_slider.value * sum(S4_sq_devs)
                )

            # End Soft Constraint

            # === Decision Strategy for Assignment Variables ===
            assign_vars = [
                _assign[(_d, _a, _s)]
                for _d in _dates
                for _a in _ambulances
                for _s in _shifts
            ]
            _model.AddDecisionStrategy(
                assign_vars, cp_model.CHOOSE_MIN_DOMAIN_SIZE, cp_model.SELECT_MIN_VALUE
            )

            # === SHOW PROBLEM COMPLEXITY ===
            complexity = compute_problem_complexity(
                _dates, _shifts, _ambulances, _shift_attrs, df_input_filtered
            )
            mo.output.replace(
                mo.vstack(
                    [
                        mo.md("🔄 **Solving...**"),
                        complexity,
                    ]
                )
            )

            # === Solve (with parallelism) ===
            if _objective_terms:
                _model.Minimize(sum(_objective_terms))
            _solver = cp_model.CpSolver()
            _solver.parameters.max_time_in_seconds = timeout_slider.value
            _solver.parameters.num_search_workers = (
                6 
            )
            _solver.parameters.log_search_progress = True

            _status = _solver.Solve(_model)

            # === EXTRACT AND DISPLAY RESULTS ===
            if _status in (cp_model.OPTIMAL, cp_model.FEASIBLE):
                _schedule_data = []
                for _d in _dates:
                    for _a in _ambulances:
                        for _s in _shifts:
                            if _solver.Value(_assign[(_d, _a, _s)]):
                                _adjustment = (
                                    unscale(_solver.Value(_hour_adjust[(_d, _a, _s)]))
                                    if _shift_attrs[_s]["flexible"]
                                    else 0
                                )
                                _actual_hours = (
                                    unscale(_shift_attrs[_s]["base_hours"]) + _adjustment
                                )
                                _schedule_data.append(
                                    {
                                        "Date": _d,
                                        "Ambulance": f"A{_a:03d}",
                                        "Shift": _s,
                                        "Base_Hours": unscale(_shift_attrs[_s][
                                            "base_hours"
                                        ]),
                                        "Hour_Adjustment": _adjustment,
                                        "Actual_Hours": _actual_hours,
                                    }
                                )
                df_schedule = pd.DataFrame(_schedule_data)
                # Typical week starts on Monday
                """ 
                df_schedule["WeekStart"] = df_schedule["Date"].apply(
                    lambda x: x - timedelta(days=x.weekday())
                )
                """
                # Rolling 7-day 
                if not df_schedule.empty:
                    _start_date = min(df_schedule["Date"])
                    df_schedule["WeekStart"] = df_schedule["Date"].apply(
                        lambda x: _start_date + timedelta(days=((x - _start_date).days // 7) * 7)
                    )

                _weekly_totals = df_schedule.groupby(["WeekStart", "Ambulance"])[
                    "Actual_Hours"
                ].sum()
                _obj_val = (
                    _solver.ObjectiveValue()
                    if _objective_terms
                    else "N/A (No soft constraints)"
                )
                schedule_result = mo.vstack(
                    [
                        mo.md(
                            f"✅ **Optimal Solution Found!** (Objective Value: {_obj_val})"
                        ),
                        mo.md("### Detailed Schedule:"),
                        mo.ui.table(df_schedule),
                        #mo.md("### Weekly Hours Verification:"),
                        #mo.ui.table(_weekly_totals.reset_index()),
                    ]
                )
            else:
                unsat_core = _solver.SufficientAssumptionsForInfeasibility()
                schedule_result = mo.md(
                    f"❌ **INFEASIBLE:** No solution exists that satisfies all hard constraints."
                )
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
def _(date, df_schedule, df_shifts, mo, pd, timedelta):
    # cell 6: Calendar View (Revised: Night Shifts, Tabs)
    if df_schedule is not None and not df_schedule.empty:
        # Dynamically determine all shift codes of type "N" from the SHIFTS sheet
        night_shift_types = [
            row["Shift"] for _, row in df_shifts.iterrows() if row["Type"] == "N"
        ]
        calendar_tabs = {}
        for _week_start, _week_group in df_schedule.groupby("WeekStart"):
            _week_group_copy = _week_group.copy()
            _week_group_copy["display_shift"] = _week_group_copy.apply(
                lambda row: f"{row['Shift']} ({row['Actual_Hours']}h)", axis=1
            )
            _calendar_df = _week_group_copy.pivot_table(
                index="Ambulance",
                columns="Date",
                values="display_shift",
                aggfunc="first",
                fill_value="",
            )
            _total_hours = _week_group.groupby("Ambulance")["Actual_Hours"].sum()
            _calendar_df["Weekly Total Hours"] = _total_hours
            # Count night shifts per ambulance (Type 'N')
            night_shift_counts = (
                _week_group[_week_group["Shift"].isin(night_shift_types)]
                .groupby("Ambulance")
                .size()
            )
            # Compute total working hours for each ambulance across the whole period
            _total_hours_whole_period = df_schedule.groupby("Ambulance")["Actual_Hours"].sum()
            _calendar_df["Total Working Hours"] = _total_hours_whole_period.reindex(_calendar_df.index)

            # Calculate number of weeks in your schedule
            _number_of_weeks = len(df_schedule["WeekStart"].unique())
            _max_total_hours = 48 * _number_of_weeks
            _hours_under_max = _max_total_hours - _total_hours_whole_period
            _calendar_df["Hours Under Max"] = _hours_under_max.reindex(_calendar_df.index)

            _calendar_df["Total Night Shifts"] = (
                night_shift_counts.reindex(_calendar_df.index)
                .fillna(0)
                .astype(int)
            )
            _calendar_df = _calendar_df.sort_index()
            _formatted_columns = {
                _col: _col.strftime("%a %m/%d")
                for _col in _calendar_df.columns
                if isinstance(_col, (pd.Timestamp, date))
            }
            _calendar_df = _calendar_df.rename(columns=_formatted_columns)
            _week_end = _week_start + timedelta(days=6)
            week_label = f"{_week_start.strftime('%Y-%m-%d')} to {_week_end.strftime('%Y-%m-%d')}"
            calendar_tabs[week_label] = mo.ui.table(_calendar_df, pagination=False)
        calendar_view = mo.ui.tabs(calendar_tabs)
    else:
        calendar_view = mo.md(
            "*Calendar view will appear after generating schedule*"
        )
    calendar_view
    return


@app.cell
def _(df_input_filtered, df_schedule, mo, pd):
    # cell 7: DAA Requirement Verification
    if (
        df_input_filtered is not None
        and df_schedule is not None
        and not df_schedule.empty
    ):
        _scheduled_counts = (
            df_schedule.groupby(["Date", "Shift"])
            .size()
            .reset_index(name="Scheduled")
        )
        _daa_summary_df = pd.merge(
            df_input_filtered, _scheduled_counts, on=["Date", "Shift"], how="left"
        )
        _daa_summary_df["Scheduled"] = (
            _daa_summary_df["Scheduled"].fillna(0).astype(int)
        )

        def _determine_status(row):
            if row["Scheduled"] == row["DAA"]:
                return "✅ Exactly Met"
            elif row["Scheduled"] > row["DAA"]:
                return "🟡 Exceeded"
            else:
                return "🔴 Missed"

        _daa_summary_df["Status"] = _daa_summary_df.apply(
            _determine_status, axis=1
        )
        _final_summary = _daa_summary_df[
            ["Date", "Shift", "DAA", "Scheduled", "Status"]
        ].rename(columns={"DAA": "Required"})
        daa_check_view = mo.vstack(
            [
                mo.md("### DAA Requirement Verification"),
                mo.ui.table(_final_summary, pagination=False),
            ]
        )
    else:
        daa_check_view = mo.md("*DAA check will appear after generating schedule*")
    daa_check_view
    return


@app.cell
def _(mo):
    export_format = mo.ui.dropdown(
        options=["XLSX", "PDF"], value="XLSX", label="Export calendar as:"
    )
    export_run_button = mo.ui.run_button(label="Export Calendar")
    return export_format, export_run_button


@app.cell
def _(
    date,
    df_schedule,
    df_shifts,
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

        # Use unique variable names for export logic
        export_night_shift_types = [
            row["Shift"] for _, row in df_shifts.iterrows() if row["Type"] == "N"
        ]

        week_tables = {}
        for _week_start, _week_group in df_schedule.groupby("WeekStart"):
            _week_group_copy = _week_group.copy()
            _week_group_copy["display_shift"] = _week_group_copy.apply(
                lambda row: f"{row['Shift']} ({row['Actual_Hours']}h)", axis=1
            )
            _calendar_df = _week_group_copy.pivot_table(
                index="Ambulance",
                columns="Date",
                values="display_shift",
                aggfunc="first",
                fill_value="",
            )

            _total_hours = _week_group.groupby("Ambulance")["Actual_Hours"].sum()

            _calendar_df["Weekly Total Hours"] = _total_hours
            # Use unique variable for night shift counts
            export_night_shift_counts = (
                _week_group[_week_group["Shift"].isin(export_night_shift_types)]
                .groupby("Ambulance")
                .size()
            )
            _total_hours_whole_period = df_schedule.groupby("Ambulance")["Actual_Hours"].sum()

            _calendar_df["Total Working Hours"] = _total_hours_whole_period.reindex(_calendar_df.index)

            # Calculate number of weeks in your schedule
            _number_of_weeks = len(df_schedule["WeekStart"].unique())
            _max_total_hours = 48 * _number_of_weeks
            _hours_under_max = _max_total_hours - _total_hours_whole_period
            _calendar_df["Hours Under Max"] = _hours_under_max.reindex(_calendar_df.index)

            _calendar_df["Total Night Shifts"] = (
                export_night_shift_counts.reindex(_calendar_df.index)
                .fillna(0)
                .astype(int)
            )
            _calendar_df = _calendar_df.sort_index()
            _formatted_columns = {
                _col: _col.strftime("%a %m/%d")
                for _col in _calendar_df.columns
                if isinstance(_col, (pd.Timestamp, date))
            }
            _calendar_df = _calendar_df.rename(columns=_formatted_columns)
            week_tables[_week_start] = _calendar_df

        if export_run_button.value:
            if export_format.value == "XLSX":
                buf = io.BytesIO()
                with pd.ExcelWriter(buf, engine="xlsxwriter") as writer:
                    for week_start, week_df in week_tables.items():
                        week_end = week_start + timedelta(days=6)
                        sheet_name = f"{week_start.strftime('%Y-%m-%d')}_to_{week_end.strftime('%m-%d')}"
                        writer.book.add_worksheet(sheet_name[:31])
                        week_df.to_excel(writer, sheet_name=sheet_name[:31])
                buf.seek(0)
                export_data = mo.download(
                    buf.getvalue(), filename="calendar.xlsx", label="Download XLSX"
                )
                export_status = mo.md(
                    "✅ XLSX calendar ready for download (one week per sheet)."
                )
            elif export_format.value == "PDF":
                try:
                    import matplotlib.pyplot as plt
                    from matplotlib.backends.backend_pdf import PdfPages

                    buf = io.BytesIO()
                    with PdfPages(buf) as pdf:
                        for week_start, week_df in week_tables.items():
                            week_end = week_start + timedelta(days=6)
                            fig, ax = plt.subplots(
                                figsize=(
                                    min(20, 2 + len(week_df.columns)),
                                    min(20, 2 + len(week_df)),
                                )
                            )
                            ax.axis("tight")
                            ax.axis("off")
                            table = ax.table(
                                cellText=week_df.values,
                                colLabels=week_df.columns,
                                rowLabels=week_df.index,
                                loc="center",
                            )
                            table.auto_set_font_size(False)
                            table.set_fontsize(8)
                            plt.title(
                                f"Week: {week_start.strftime('%Y-%m-%d')} to {week_end.strftime('%Y-%m-%d')}"
                            )
                            pdf.savefig(fig, bbox_inches="tight")
                            plt.close(fig)
                    buf.seek(0)
                    export_data = mo.download(
                        buf.getvalue(),
                        filename="calendar.pdf",
                        label="Download PDF",
                    )
                    export_status = mo.md(
                        "✅ PDF calendar ready for download (one week per page)."
                    )
                except ImportError:
                    export_status = mo.md(
                        "❌ PDF export requires matplotlib. Please install it in your environment."
                    )

        export_controls = mo.vstack(
            [
                mo.hstack([export_format, export_run_button]),
                export_status,
                export_data if export_data else mo.md(""),
            ]
        )
    else:
        export_controls = mo.md("*No schedule to export yet.*")

    export_controls
    return


@app.cell
def _(mo):
    # cell 9: Complexity Metric Function


    def compute_problem_complexity(
        _dates, _shifts, _ambulances, _shift_attrs, df_input
    ):
        # Assignment variables
        num_assign_vars = len(_dates) * len(_ambulances) * len(_shifts)

        # Hard constraints
        num_hard_demand_constraints = len(_dates) * (len(_shifts) - 1)
        num_hard_one_shift_constraints = len(_dates) * len(_ambulances)
        num_night_shifts = sum(
            1 for s in _shifts if _shift_attrs[s]["type"] == "N"
        )
        num_hard_night_rest_constraints = (
            len(_ambulances) * (len(_dates) - 1) * num_night_shifts
        )
        num_weeks = len(_dates) // 7
        num_hard_weekly_hours_constraints = len(_ambulances) * num_weeks
        num_hard_constraints = (
            num_hard_demand_constraints
            + num_hard_one_shift_constraints
            + num_hard_night_rest_constraints
            + num_hard_weekly_hours_constraints
        )

        # Soft constraints
        num_soft_fairness_constraints = len(_ambulances) * 3
        num_day_shift_types = sum(
            1 for s in _shifts if _shift_attrs[s]["type"] == "D"
        )
        num_soft_consistency_constraints = len(_ambulances) * num_day_shift_types
        num_soft_constraints = (
            num_soft_fairness_constraints + num_soft_consistency_constraints
        )

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
