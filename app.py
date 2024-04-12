from omnidata_daily import DATA
from omnidata_hourly import H_DATA
from flask import Flask, render_template, request


app = Flask(__name__)

@app.route("../templates/index.html", methods=["POST", "GET"])
def index():
    if request.method == "POST":
        startYear = request.form.get("startYear")
        endYear = request.form.get("endYear")
        result = DATA(startYear,endYear)
    else:
        return render_template(index.html, result=result.print_year_day())

if __name__ == '__main__':
    app.run(debug=True)
        