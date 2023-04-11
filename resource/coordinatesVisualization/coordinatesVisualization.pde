// The following short CSV file called "mammals.csv" is parsed
// in the code below. It must be in the project's "data" folder.
//
// id,species,name
// 0,Capra hircus,Goat
// 1,Panthera pardus,Leopard
// 2,Equus zebra,Zebra

Table table;
FloatList y = new FloatList();
FloatList x = new FloatList();
FloatList s = new FloatList();

int rowCount = 0;
int columnCount = 0;
int landmark = 0;


void setup() {

  size(1920, 1080);
  //background(0);

  table = loadTable("../../data/csv/swingSingle.csv", "header");

  //println(table.getRowCount() + " total rows in table");
  //println(table);
  table.removeColumn(0); //Remove time column
}

void draw() {
  background(0);

  ///////
  TableRow row = table.getRow(rowCount);

  for (int column = 0; column < row.getColumnCount(); column++) {
    //print(row.getFloat(column));
    //print(", ");
    if (column%3 == 0) {
      y.append(row.getFloat(column));
    } else if (column%3 == 1) {
      x.append(row.getFloat(column));
    } else if (column%3 == 2) {
      s.append(row.getFloat(column));
    }
  }

  // Draw Line
  int strokeAlpha = 125;
  strokeWeight(10);
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
    circle(x.get(i)*1980, y.get(i)*1080, 20);
  }




  delay(100);

  y.clear();
  x.clear();
  s.clear();

  rowCount++;
  if (rowCount == table.getRowCount()) {
    exit();
  }
}

// Sketch prints:
// 3 total rows in table
// Goat (Capra hircus) has an ID of 0
// Leopard (Panthera pardus) has an ID of 1
// Zebra (Equus zebra) has an ID of 2

//for (TableRow row : table.rows()) {
//  //background(0);
//  //y = new FloatList();
//  //x = new FloatList();
//  //s = new FloatList();

//  for (int column = 0; column < row.getColumnCount(); column++) {
//    //print(row.getFloat(column));
//    //print(", ");
//    if (column%3 == 0) {
//      y.append(row.getFloat(column));
//    } else if (column%3 == 1) {
//      x.append(row.getFloat(column));
//    } else if (column%3 == 2) {
//      s.append(row.getFloat(column));
//    }
//  }
//  //println("");
//  //println("=========");
//  println(y.get(0));

//  for (int i = 0; i <17; i++) {
//    fill(255, 0, 0);
//    circle(x.get(i)*1000, y.get(i)*1000, 10);
//  }
//  delay(1);

//  y.clear();
//  x.clear();
//  s.clear();
//}
