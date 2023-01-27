<!DOCTYPE html>
<html>
    <head>
        <meta charset="UTF-8">
        <title>Add New Employee</title>
    </head>
    <body>
        <h2>Add New Employee</h2>
        <a href="cao_yiting_IP_Task7_home.jsp">Back to Home Page</a>
        <!--
            Form for collecting user input for the new employee record.
            Upon form submission, cao_yiting_IP_Task7_query1_success.jsp file will be invoked.
        -->
        <form action="cao_yiting_IP_Task7_query1_success.jsp">
            <!-- The form organized in an HTML table for better clarity. -->
            <table border=1>
                <tr>
                    <th colspan="2">Enter a new employee. Provide employee name, address, salary, employee type (worker/quality controller/technical staff), and type specific parameter(s) (worker: maximum number of product per day; quality controller: product type; technical staff: technical position, bachelor's degree, master's degree, phd degree [omit that entry if none]).</th>
                </tr>
                <tr>
                    <td>Employee Name:</td>
                    <td><div style="text-align: center;">
                    <input type=text name=ename>
                    </div></td>
                </tr>
                <tr>
                    <td>Employee Address:</td>
                    <td><div style="text-align: center;">
                    <input type=text name=address>
                    </div></td>
                </tr>
                <tr>
                    <td>Employee Salary:</td>
                    <td><div style="text-align: center;">
                    <input type=text name=salary>
                    </div></td>
                </tr>
                <tr>
                    <td>Employee Type:</td>
                    <td><div style="text-align: center;">
                    <input type=text name=type>
                    </div></td>
                </tr>
                <tr>
                    <td>Parameter 1:</td>
                    <td><div style="text-align: center;">
                    <input type=text name=param1>
                    </div></td>
                </tr>
                <tr>
                    <td>Bachelor's Degree:</td>
                    <td><div style="text-align: center;">
                    <input type=text name=param2a>
                    </div></td>
                </tr>
                <tr>
                    <td>Master's Degree:</td>
                    <td><div style="text-align: center;">
                    <input type=text name=param2b>
                    </div></td>
                </tr>
                <tr>
                    <td>PhD Degree:</td>
                    <td><div style="text-align: center;">
                    <input type=text name=param2c>
                    </div></td>
                </tr>
                <tr>
                    <td><div style="text-align: center;">
                    <input type=reset value=Clear>
                    </div></td>
                    <td><div style="text-align: center;">
                    <input type=submit value=Insert>
                    </div></td>
                </tr>
            </table>
        </form>
    </body>
</html>
