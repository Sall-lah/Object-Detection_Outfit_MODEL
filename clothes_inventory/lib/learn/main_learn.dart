import 'package:flutter/material.dart';

// Main Widget Tree
void main() {
  runApp(MaterialApp(home: LeTachyon()));
}

// // Stateless
// class MyApp extends StatelessWidget {
//   const MyApp({super.key});

//   @override
//   Widget build(BuildContext context) {
//     return Scaffold(
//       backgroundColor: Colors.white,
//       appBar: AppBar(
//         title: Text(
//           'Clothes Inventory',
//           style: TextStyle(
//             fontFamily: 'Montserrat',
//             fontWeight: FontWeight.bold,
//             color: Color.fromARGB(255, 0, 0, 0),
//           ),
//         ),
//         backgroundColor: Colors.greenAccent,
//       ),
//       // body: Center(
//       //   // Ver 1.
//       //   // child: Image(
//       //   //   // image: NetworkImage('https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcRvLcIMBXDmHax_aP15E-l20phAXaxjbimGZQ&s'), // Network Image
//       //   //   image: AssetImage('assets/agnes.png'), // Asset Image
//       //   // ),
//       //   // Ver 2.
//       //   // child: Image.asset('assets/agnes.png'),
//       //   // Button
//       //   child: ElevatedButton.icon(
//       //     style: ElevatedButton.styleFrom(
//       //       backgroundColor: Colors.greenAccent,
//       //       foregroundColor: const Color.fromARGB(255, 0, 0, 0),
//       //       ),
//       //     // // StateFull
//       //     // style: ButtonStyle(
//       //     //   foregroundColor: WidgetStateProperty.all<Color>(const Color.fromARGB(255, 0, 0, 0)),
//       //     //   overlayColor: WidgetStateProperty.all<Color>(Colors.greenAccent),
//       //     //   ),
//       //     onPressed: () {
//       //       print("You clicked me!");
//       //       },
//       //     icon: Icon(Icons.add_a_photo),
//       //     label: Text('Add Photo'),
//       //   )
//       // ),

//       // // Container
//       // body: Container(
//       //   // padding: EdgeInsets.all(30),
//       //   margin: EdgeInsets.all(15),
//       //   alignment: Alignment.center,
//       //   color: Colors.greenAccent[100],
//       //   child: Padding(
//       //     padding: EdgeInsets.all(20),
//       //     child: Text(
//       //       'Umadachi'
//       //     ),
//       //   ),
//       //   // child: Text(
//       //   //   'Umadachi'
//       //   // ),
//       // ),

//       // // Column and Row (alike)
//       // body: Column(
//       //   mainAxisAlignment: MainAxisAlignment.center,
//       //   // crossAxisAlignment: CrossAxisAlignment.end,
//       //   children: <Widget>[
//       //     Container(
//       //       margin: EdgeInsets.all(15),
//       //       alignment: Alignment(-0.5, 0), // Use all of the available space
//       //       color: Colors.greenAccent[400],
//       //       child: Text("Hey"),
//       //     ),
//       //     Container(
//       //       margin: EdgeInsets.all(15),
//       //       alignment: Alignment.center, // Use all of the available space
//       //       color: Colors.greenAccent[100],
//       //       child: Text("Hey"),
//       //     ),
//       //     Center( // Only use the space that is needed
//       //       child: Container(
//       //         margin: EdgeInsets.all(15),
//       //         color: Colors.greenAccent[200],
//       //         child: Text("There"),
//       //       ),
//       //     ),
//       //   ],
//       // ),

//       // body: Row(
//       //   mainAxisAlignment: MainAxisAlignment.center,
//       //   children: <Widget>[
//       //     Expanded( // memakan tempat yang tersedia
//       //       flex: 3, // 
//       //       child: Container(
//       //         padding: EdgeInsets.all(30),
//       //         color: Colors.cyan,
//       //         child: Text("1"),
//       //       ),
//       //     ),
//       //     Expanded(
//       //       flex: 2,
//       //       child: Container(
//       //         padding: EdgeInsets.all(30),
//       //         color: Colors.red,
//       //         child: Text("2"),
//       //       ),
//       //     ),
//       //     Expanded(
//       //       flex: 1,
//       //       child: Container(
//       //         padding: EdgeInsets.all(30),
//       //         color: Colors.yellow,
//       //         child: Text("3"),
//       //       ),
//       //     ),
//       //   ],
//       // ),

//       body: Padding(
//         padding: const EdgeInsets.all(8.0),
//         child: Column(
//           children: [
//             Row(
//               mainAxisAlignment: MainAxisAlignment.center, // Center the row ( unnescessary since we use expanded and flex inside )
//               children: [
//                 Expanded(// use all available space
//                   flex: 1,
//                   child: Container(
//                     alignment: Alignment.center, // center the image Container
//                     padding: EdgeInsets.all(10),
//                     height: 200,
//                     color: const Color.fromARGB(255, 167, 255, 4),
//                     child: Image.asset('assets/dum.jpg'),
//                   ),
//                 ),
//                 // SizedBox(width: 10,), // make space between the containers 
//                 Expanded(// use all available space
//                   flex: 1,
//                   child: Container(
//                     alignment: Alignment.center, // center the image Container
//                     padding: EdgeInsets.all(10),
//                     height: 200,
//                     color: Colors.cyan,
//                     child: Image.asset('assets/agnes.png'),
//                   ),
//                 ),
//               ],
//             ),
//           ],
//         ),
//       ),

//       floatingActionButton: FloatingActionButton.extended(
//         backgroundColor: Colors.greenAccent,
//         onPressed: () {
//           print("Yeppers");
//         },
//         // shape: const CircleBorder(),
//         icon: const Icon(Icons.add_a_photo, size: 30, color: Color.fromARGB(255, 0, 0, 0)),
//         label: const Text('Add Photo', style: TextStyle(color: Color.fromARGB(255, 0, 0, 0)),),
//         // child: const Icon(Icons.add_a_photo, size: 30, color: Color.fromARGB(255, 0, 0, 0)),
//       ),
//     );
//   }
// }


// Statefull
class LeTachyon extends StatefulWidget {
  const LeTachyon({super.key});

  @override
  State<LeTachyon> createState() => _LeTachyonState();
}

class _LeTachyonState extends State<LeTachyon> {

  double size = 100;

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      backgroundColor: Colors.white,
      appBar: AppBar(
        title: Text(
          'Le Tachyon',
          style: TextStyle(
            fontFamily: 'Montserrat',
            fontWeight: FontWeight.bold,
            color: Color.fromARGB(255, 0, 0, 0),
          ),
        ),
        backgroundColor: Colors.greenAccent,
      ),

      body: Container(
        // color: Colors.greenAccent[100],
        alignment: Alignment.center, // center the image Container
        child: Image.asset('assets/agnes.png', height: size, fit: BoxFit.cover),
      ),

      floatingActionButton: FloatingActionButton.extended(
        backgroundColor: Colors.greenAccent,
        onPressed: () {
          setState(() {
            size += 100.0;
          });
        },
        // icon: const Icon(Icons.add_circle_outline, size: 30, color: Color.fromARGB(255, 0, 0, 0)),
        label: const Text('Increase Tachyon', style: TextStyle(color: Color.fromARGB(255, 0, 0, 0)),),
      ),
    );
  }
}
