<%@ page language="java" contentType="text/html; charset=UTF-8"
pageEncoding="UTF-8"%>
<!DOCTYPE html PUBLIC "-//W3C//DTD HTML 4.01 Transitional//EN"
"http://www.w3.org/TR/html4/loose.dtd">
<html>
<head>
<meta http-equiv="Content-Type" content="text/html; charset=UTF-8">
<title>Query Result</title>
</head>
    <body>
    <%@page import="cao_yiting_IP_Task7.cao_yiting_IP_Task7_DataHandler"%>
    <%@page import="java.sql.ResultSet"%>
    <%@page import="java.sql.Array"%>
    <%
    // The handler is the one in charge of establishing the connection.
    cao_yiting_IP_Task7_DataHandler handler = new cao_yiting_IP_Task7_DataHandler();

    // Get the attribute values passed from the input form.
    String ename = request.getParameter("ename");
    String address = request.getParameter("address");
    String salary = request.getParameter("salary");
    String type = request.getParameter("type");
    String p1 = request.getParameter("param1");
    String p2a = request.getParameter("param2a");
    String p2b = request.getParameter("param2b");
    String p2c = request.getParameter("param2c");

    /*
     * If the user hasn't filled out all the required parameters. This is very simple checking.
     */
    if (ename.equals("") || address.equals("") || salary.equals("") || type.equals("") || p1.equals("")) {
        response.sendRedirect("add_movie_form.jsp");
    } else {
        
        // Now perform the query with the data from the form.
        boolean success = handler.addEmployee(ename, address, salary, type, p1, p2a, p2b, p2c);
        if (!success) { // Something went wrong
            %>
                <h2>There was a problem inserting </h2>
            <%
        } else { // Confirm success to the user
            %>
            <h2>The Employee:</h2>

            <ul>
                <li>Name: <%=ename%></li>
                <li>Address: <%=address%></li>
                <li>Salary: <%=salary%></li>
                <li>Type: <%=type%></li>
                <li>Parameter 1: <%=p1%></li>
                <li>Parameter 2a: <%=p2a%></li>
                <li>Parameter 2b: <%=p2b%></li>
                <li>Parameter 2c: <%=p2c%></li>
            </ul>

            <h2>Was successfully inserted.</h2>
            
            <a href="cao_yiting_IP_Task7_home.jsp">Back to Home Page.</a>
            <%
        }
    }
    %>
    </body>
</html>
