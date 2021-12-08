import javax.imageio.ImageIO;
import java.awt.*;
import java.awt.image.BufferedImage;
import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileWriter;
import java.io.IOException;
import java.net.URL;
import java.util.Scanner;
import java.util.concurrent.Executors;
import java.util.concurrent.ThreadPoolExecutor;
import java.util.concurrent.TimeUnit;

public class Main {
    final static double tileWidthMeters = 80.0;
    final static int tileWidthPixels = 256;

//    public static void main(String[] args) {
//        try {
//            File myObj = new File("swamp_forest_woods_grid.txt");
//            Scanner myReader = new Scanner(myObj);
//            File lastI = new File("lastI.txt");
//            Scanner scannerI = new Scanner(lastI);
//            int lastIInt = scannerI.nextInt();
//            int i = 0;
//            while (myReader.hasNextLine()) {
//                ThreadPoolExecutor executor = (ThreadPoolExecutor) Executors.newCachedThreadPool();
//                String[] datasplit1, datasplit2, datasplit3, datasplit4, datasplit5, datasplit6, datasplit7, datasplit8;
//
//                datasplit1 = myReader.nextLine().split(",");
//                datasplit2 = myReader.nextLine().split(",");
//                datasplit3 = myReader.nextLine().split(",");
//                datasplit4 = myReader.nextLine().split(",");
//                datasplit5 = myReader.nextLine().split(",");
//                datasplit6 = myReader.nextLine().split(",");
//                datasplit7 = myReader.nextLine().split(",");
//                datasplit8 = myReader.nextLine().split(",");
//                if (i <= lastIInt) {
//                    i++;
//                    continue;
//                }
//                int i1 = 8*i;
//                int i2 = 8*i+1;
//                int i3 = 8*i+2;
//                int i4 = 8*i+3;
//                int i5 = 8*i+4;
//                int i6 = 8*i+5;
//                int i7 = 8*i+6;
//                int i8 = 8*i+7;
//                downloadAndSave(executor, datasplit1, i1);
//                downloadAndSave(executor, datasplit2, i2);
//                downloadAndSave(executor, datasplit3, i3);
//                downloadAndSave(executor, datasplit4, i4);
//                downloadAndSave(executor, datasplit5, i5);
//                downloadAndSave(executor, datasplit6, i6);
//                downloadAndSave(executor, datasplit7, i7);
//                downloadAndSave(executor, datasplit8, i8);
//                executor.shutdown();
//                try {
//                    executor.awaitTermination(3600, TimeUnit.SECONDS);
//                } catch (InterruptedException ignored) {}
//                i++;
//                FileWriter f = new FileWriter("lastI.txt");
//                f.write(i + "");
//                f.close();
//            }
//            myReader.close();
//        } catch (FileNotFoundException e) {
//            System.out.println("An error occurred.");
//            e.printStackTrace();
//        } catch (IOException e) {
//            e.printStackTrace();
//        }
//    }

    // Downloads 8 tiles at a time, should be about 12 tiles per second, -> about 24 hours of downloading lol.
    public static void main(String[] args) {
        try {
            File myObj = new File("mlpoints1.txt");
            Scanner myReader = new Scanner(myObj);
            File lastI = new File("lastI.txt");
            Scanner scannerI = new Scanner(lastI);
            int lastIInt = scannerI.nextInt();
            int i = 0;
            while (myReader.hasNextLine()) {
                ThreadPoolExecutor executor = (ThreadPoolExecutor) Executors.newCachedThreadPool();
                String[] datasplit1, datasplit2, datasplit3, datasplit4, datasplit5, datasplit6, datasplit7, datasplit8;

                datasplit1 = myReader.nextLine().split(",");
                datasplit2 = myReader.nextLine().split(",");
                datasplit3 = myReader.nextLine().split(",");
                datasplit4 = myReader.nextLine().split(",");
                datasplit5 = myReader.nextLine().split(",");
                datasplit6 = myReader.nextLine().split(",");
                datasplit7 = myReader.nextLine().split(",");
                datasplit8 = myReader.nextLine().split(",");
                if (i <= lastIInt) {
                    i++;
                    continue;
                }
                int i1 = 8*i;
                int i2 = 8*i+1;
                int i3 = 8*i+2;
                int i4 = 8*i+3;
                int i5 = 8*i+4;
                int i6 = 8*i+5;
                int i7 = 8*i+6;
                int i8 = 8*i+7;
                downloadAndSave(executor, datasplit1, i1);
                downloadAndSave(executor, datasplit2, i2);
                downloadAndSave(executor, datasplit3, i3);
                downloadAndSave(executor, datasplit4, i4);
                downloadAndSave(executor, datasplit5, i5);
                downloadAndSave(executor, datasplit6, i6);
                downloadAndSave(executor, datasplit7, i7);
                downloadAndSave(executor, datasplit8, i8);
                executor.shutdown();
                try {
                    executor.awaitTermination(3600, TimeUnit.SECONDS);
                } catch (InterruptedException ignored) {}
                i++;
                FileWriter f = new FileWriter("lastI.txt");
                f.write(i + "");
                f.close();
            }
            myReader.close();
        } catch (FileNotFoundException e) {
            System.out.println("An error occurred.");
            e.printStackTrace();
        } catch (IOException e) {
            e.printStackTrace();
        }
    }
//    public static void main(String[] args) {
//        // Convert to jpg
//        // TOP LEFT IS 34.4839943, -78.636374
//        // BOTTOM RIGHT IS 35.9329639, -78.5666948
//
//        for (int i = 0; i < 1024; i+=128) {
//            double wt = i/128.0/128.0;
//            Image img = downloadImagery(wt*34.4770182 + (1-wt)*34.4839943, -78.636374);
//            saveToFile(img, "trainingData/images3/img_"+i+".jpg");
//        }
//    }

//    public static void main(String[] args) {
//        for (int i = 0; i < 506316; i++) {
//            // add 599290 to each image and save to training data 2.
//            if (Math.random() < 0.95) continue;
//            try {
//                File f1 = new File("trainingData/woods/train_"+i+".jpg");
//                Image img = ImageIO.read(f1);
//                saveToFile(img, "trainingData/newWoods/train_"+(i)+".jpg");
//                if (i % 100 == 0) System.out.println(i + " in woods    finished.");
//            } catch (IOException e) {
//            }
//            try {
//                File f1 = new File("trainingData/notWoods/train_"+i+".jpg");
//                Image img = ImageIO.read(f1);
//                saveToFile(img, "trainingData/newNotWoods/train_"+(i)+".jpg");
//                if (i % 100 == 0) System.out.println(i + " in notWoods finished.");
//            } catch (IOException e) {
//            }
//        }
//    }

    private static void downloadAndSave(ThreadPoolExecutor executor, String[] datasplit, int i) {
        executor.execute(() -> {
            Image img = downloadImagery(Double.parseDouble(datasplit[0]), Double.parseDouble(datasplit[1]));
            if (i%8 == 0) System.out.println(i);
            saveToFile(img, (datasplit[2].equals("1") ? "trainingData/edgeCaseWoods/" : "trainingData/edgeCaseNotWoods/") + "train_" + i + ".jpg");
        });
    }

    private static Image downloadImagery(double lat, double lon) {
        // Format: path1 + width + path2 + height + path3 + bbox + path4
        // Bbox looks like "-8770461.06124855,4287887.9751166515,-8770384.624220276,4287964.412144927" but no ""
        double metersPerDegreeNorthSouth = 40007863.0/180.0;
        double metersPerDegreeEastWest = Math.cos(lat/180.0*Math.PI)*metersPerDegreeNorthSouth;
        double left = lon-tileWidthMeters/2/metersPerDegreeEastWest;
        double up = lat+tileWidthMeters/2/metersPerDegreeNorthSouth;
        double right = lon+tileWidthMeters/2/metersPerDegreeEastWest;
        double down = lat-tileWidthMeters/2/metersPerDegreeNorthSouth;
        latlon topleft = toMeterProjection(new latlon(up, left));
        latlon bottomright = toMeterProjection(new latlon(down, right));
        String path = "https://services.nconemap.gov/secure/services/Imagery/Orthoimagery_2017" +
                "/ImageServer/WMSServer?LAYERS=0&STYLES=&FORMAT=image/jpeg&CRS=EPSG:3857&WIDTH="
                + tileWidthPixels + "&HEIGHT=" + tileWidthPixels + "&BBOX=" +
                topleft.lon + "," + bottomright.lat + "," + bottomright.lon + "," + topleft.lat +
                "&VERSION=1.3.0&SERVICE=WMS&REQUEST=GetMap";
        System.out.println(path);
        try {
            return ImageIO.read(new URL(path));
        } catch (IOException e) {
            // handle IOException
            return null;
        }
    }

    public static void saveToFile(Image image, String path) {
        File outputFile = new File(path);
        BufferedImage bImage = toBufferedImage(image);
        try {
            ImageIO.write(bImage, "jpg", outputFile);
        } catch (IOException e) {
            throw new RuntimeException(e);
        }
    }

    public static BufferedImage toBufferedImage(Image img) {
        if (img instanceof BufferedImage) {
            return (BufferedImage) img;
        }

        // Create a buffered image with transparency
        BufferedImage bimage = new BufferedImage(img.getWidth(null), img.getHeight(null), BufferedImage.TYPE_INT_ARGB);

        // Draw the image on to the buffered image
        Graphics2D bGr = bimage.createGraphics();
        bGr.drawImage(img, 0, 0, null);
        bGr.dispose();

        // Return the buffered image
        return bimage;
    }

    public static latlon toMeterProjection(latlon in) {
        double lonInEPSG4326 = in.lon;
        double latInEPSG4326 = in.lat;

        double lonInEPSG3857 = (lonInEPSG4326 * 20037508.34 / 180);
        double latInEPSG3857 = (Math.log(Math.tan((90 + latInEPSG4326) * Math.PI / 360)) / (Math.PI / 180)) * (20037508.34 / 180);
        return new latlon(latInEPSG3857, lonInEPSG3857);
    }

}

class latlon {
    double lat, lon;
    public latlon(double lat, double lon) {
        this.lat = lat;
        this.lon = lon;
    }
}
