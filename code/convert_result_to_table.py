import pandas as pd

# 1. Read your CSV
df = pd.read_csv("battery_combined_100_005_100_capacity.csv")

# 2. Pivot so that each country is a row, and you get four metrics per method
pivot = df.pivot(
    index="country",
    columns="Method",
    values=["Objective", "Num zeros in y"]
)

# 3. Flatten the column MultiIndex
pivot.columns = ["_".join(col).strip() for col in pivot.columns]

# 4. Rename only the “combined” columns to your desired final names
pivot = pivot.rename(columns={
    "Objective_wind only":        "Wind Objective",
    "Objective_pv only":          "PV Objective",
    "Objective_combined":         "Combined Objective",
    "Num zeros in y_combined":    "Num zeros in y"
})

# 5. Compute the synergy ratio = Combined / (Wind + PV)
pivot["Synergy Ratio"] = (
    pivot["Combined Objective"]
    / (pivot["Wind Objective"] + pivot["PV Objective"])
)

# 6. Re‐order & round to two decimals
final = pivot[
    [
        "Wind Objective",
        "PV Objective",
        "Combined Objective",
        "Synergy Ratio",
        "Num zeros in y",
    ]
].round(2)

# 7. Write to Excel with formatting
with pd.ExcelWriter("results_battery_100_5_100_capacity.xlsx", engine="xlsxwriter") as writer:
    final.to_excel(writer, sheet_name="Profits", startrow=1, header=False)

    workbook  = writer.book
    worksheet = writer.sheets["Profits"]

    # Header formatting
    header_fmt = workbook.add_format({
        "bold":   True,
        "align":  "center",
        "border": 1
    })
    for col_num, value in enumerate(final.columns.insert(0, "Country")):
        worksheet.write(0, col_num, value, header_fmt)

    # Turn into an Excel table
    (max_row, max_col) = final.shape
    worksheet.add_table(
        0, 0,
        max_row, max_col,
        {
            "columns": [{"header": h} for h in ["Country"] + final.columns.tolist()],
            "style":   "Table Style Medium 9"
        }
    )

    # Number formats
    num_fmt = workbook.add_format({"num_format": "0.00"})
    int_fmt = workbook.add_format({"num_format": "0"})
    worksheet.set_column(1, 3, 12, num_fmt)  # objectives
    worksheet.set_column(4, 4, 12, num_fmt)  # ratio
    worksheet.set_column(5, 5, 10, int_fmt)  # zeros
    # worksheet.set_column(6, 7, 12, num_fmt)  # p_max, q_max
