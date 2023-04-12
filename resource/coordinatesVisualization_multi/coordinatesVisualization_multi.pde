// The following short CSV file called "mammals.csv" is parsed
// in the code below. It must be in the project's "data" folder.
//
// id,species,name
// 0,Capra hircus,Goat
// 1,Panthera pardus,Leopard
// 2,Equus zebra,Zebra

Table table;
FloatList y0 = new FloatList();
FloatList x0 = new FloatList();
FloatList s0 = new FloatList();
FloatList y1 = new FloatList();
FloatList x1 = new FloatList();
FloatList s1 = new FloatList();
FloatList y2 = new FloatList();
FloatList x2 = new FloatList();
FloatList s2 = new FloatList();
FloatList y3 = new FloatList();
FloatList x3 = new FloatList();
FloatList s3 = new FloatList();
FloatList y4 = new FloatList();
FloatList x4 = new FloatList();
FloatList s4 = new FloatList();
FloatList y5 = new FloatList();
FloatList x5 = new FloatList();
FloatList s5 = new FloatList();


int rowCount = 0;
//int columnCount = 0;
//int landmark = 0;


void setup() {

  size(1950, 1080);
  //background(0);

  table = loadTable("../../data/csv/temporalEasy.csv", "header");

  //println(table.getRowCount() + " total rows in table");
  //println(table);
  table.removeColumn(0); //Remove time column
}

void draw() {
  background(0);

  ///////
  TableRow row = table.getRow(rowCount);


  //for (int column = 0; column < row.getColumnCount(); column++) {
  //  //print(row.getFloat(column));
  //  //print(", ");
  //  if (column%3 == 0) {
  //    y.append(row.getFloat(column));
  //  } else if (column%3 == 1) {
  //    x.append(row.getFloat(column));
  //  } else if (column%3 == 2) {
  //    s.append(row.getFloat(column));
  //  }
  //}

  for (int column = 0; column < row.getColumnCount(); column++) {
    if (column/51 == 0) {
      if (column%3 == 0) {
        y0.append(row.getFloat(column));
      } else if (column%3 == 1) {
        x0.append(row.getFloat(column));
      } else if (column%3 == 2) {
        s0.append(row.getFloat(column));
      }
    } else if (column/51 == 1) {
      if (column%3 == 0) {
        y1.append(row.getFloat(column));
      } else if (column%3 == 1) {
        x1.append(row.getFloat(column));
      } else if (column%3 == 2) {
        s1.append(row.getFloat(column));
      }
    } else if (column/51 == 2) {
      if (column%3 == 0) {
        y2.append(row.getFloat(column));
      } else if (column%3 == 1) {
        x2.append(row.getFloat(column));
      } else if (column%3 == 2) {
        s2.append(row.getFloat(column));
      }
    } else if (column/51 == 3) {
      if (column%3 == 0) {
        y3.append(row.getFloat(column));
      } else if (column%3 == 1) {
        x3.append(row.getFloat(column));
      } else if (column%3 == 2) {
        s3.append(row.getFloat(column));
      }
    } else if (column/51 == 4) {
      if (column%3 == 0) {
        y4.append(row.getFloat(column));
      } else if (column%3 == 1) {
        x4.append(row.getFloat(column));
      } else if (column%3 == 2) {
        s4.append(row.getFloat(column));
      }
    } else if (column/51 == 5) {
      if (column%3 == 0) {
        y5.append(row.getFloat(column));
      } else if (column%3 == 1) {
        x5.append(row.getFloat(column));
      } else if (column%3 == 2) {
        s5.append(row.getFloat(column));
      }
    }
  }
  print(x0);

  visualize(x0, y0, s0);
  visualize(x1, y1, s1);
  visualize(x2, y2, s2);
  visualize(x3, y3, s3);
  visualize(x4, y4, s4);
  visualize(x5, y5, s5);

  delay(50);

  makeClear(x0, y0, s0);
  makeClear(x1, y1, s1);
  makeClear(x2, y2, s2);
  makeClear(x3, y3, s3);
  makeClear(x4, y4, s4);
  makeClear(x5, y5, s5);

  rowCount++;
  if (rowCount == table.getRowCount()) {
    exit();
  }
}

void visualize(FloatList x, FloatList y, FloatList s) {
  // Draw Line
  int strokeAlpha = 125;
  strokeWeight(5);
  stroke(255-(255*(s.get(5)+s.get(5))/2), (255*(s.get(6)+s.get(6))/2), 0, strokeAlpha);
  line(x.get(5)*1980, y.get(5)*1080, x.get(6)*1980, y.get(6)*1080);
  stroke(255-(255*(s.get(5)+s.get(5))/2), (255*(s.get(7)+s.get(7))/2), 0, strokeAlpha);
  line(x.get(5)*1980, y.get(5)*1080, x.get(7)*1980, y.get(7)*1080);
  stroke(255-(255*(s.get(5)+s.get(5))/2), (255*(s.get(11)+s.get(11))/2), 0, strokeAlpha);
  line(x.get(5)*1980, y.get(5)*1080, x.get(11)*1980, y.get(11)*1080);
  stroke(255-(255*(s.get(6)+s.get(6))/2), (255*(s.get(8)+s.get(8))/2), 0, strokeAlpha);
  line(x.get(6)*1980, y.get(6)*1080, x.get(8)*1980, y.get(8)*1080);
  stroke(255-(255*(s.get(6)+s.get(6))/2), (255*(s.get(12)+s.get(12))/2), 0, strokeAlpha);
  line(x.get(6)*1980, y.get(6)*1080, x.get(12)*1980, y.get(12)*1080);
  stroke(255-(255*(s.get(7)+s.get(7))/2), (255*(s.get(9)+s.get(9))/2), 0, strokeAlpha);
  line(x.get(7)*1980, y.get(7)*1080, x.get(9)*1980, y.get(9)*1080);
  stroke(255-(255*(s.get(8)+s.get(8))/2), (255*(s.get(10)+s.get(10))/2), 0, strokeAlpha);
  line(x.get(8)*1980, y.get(8)*1080, x.get(10)*1980, y.get(10)*1080);
  stroke(255-(255*(s.get(11)+s.get(11))/2), (255*(s.get(12)+s.get(12))/2), 0, strokeAlpha);
  line(x.get(11)*1980, y.get(11)*1080, x.get(12)*1980, y.get(12)*1080);
  stroke(255-(255*(s.get(11)+s.get(11))/2), (255*(s.get(13)+s.get(13))/2), 0, strokeAlpha);
  line(x.get(11)*1980, y.get(11)*1080, x.get(13)*1980, y.get(13)*1080);
  stroke(255-(255*(s.get(12)+s.get(12))/2), (255*(s.get(14)+s.get(14))/2), 0, strokeAlpha);
  line(x.get(12)*1980, y.get(12)*1080, x.get(14)*1980, y.get(14)*1080);
  stroke(255-(255*(s.get(13)+s.get(13))/2), (255*(s.get(15)+s.get(15))/2), 0, strokeAlpha);
  line(x.get(13)*1980, y.get(13)*1080, x.get(15)*1980, y.get(15)*1080);
  stroke(255-(255*(s.get(14)+s.get(14))/2), (255*(s.get(16)+s.get(16))/2), 0, strokeAlpha);
  line(x.get(14)*1980, y.get(14)*1080, x.get(16)*1980, y.get(16)*1080);

  // Draw Coordinates
  for (int i = 0; i <17; i++) {
    noStroke();
    fill(255-255*s.get(i), 255*s.get(i), 0, 180);
    circle(x.get(i)*1980, y.get(i)*1080, 10);
  }
}

void makeClear(FloatList y, FloatList x, FloatList s) {
  y.clear();
  x.clear();
  s.clear();
}
