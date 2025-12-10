import 'package:flutter/material.dart';
import 'card.dart';

void main() {
  runApp(MaterialApp(home: myApp()));
}

class myApp extends StatefulWidget {
  const myApp({super.key});

  @override
  State<myApp> createState() => _myAppState();
}

class _myAppState extends State<myApp> {


  
  List<NameCard> nameCard = [
    NameCard(path: 'assets/agnes.png', name: 'Agnes'),
    NameCard(path: 'assets/agnes.png', name: 'Agnes'),
    NameCard(path: 'assets/dum.jpg', name: 'Agnes')
  ];

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      backgroundColor: Colors.white,
      appBar: AppBar(
        title: Text(
          'Clothes Inventory',
          style: TextStyle(
            fontFamily: 'Montserrat',
            fontWeight: FontWeight.bold,
            color: Color.fromARGB(255, 0, 0, 0),
          ),
        ),
        backgroundColor: Colors.greenAccent,
      ),

      // // Ver 1.
      // body: Column(
      //   children: nameCard.map((i) {
      //     return Expanded(
      //       flex: 1,
      //       child: Column(
      //         children: [
      //           Flexible(
      //             flex: 3,
      //             child: Container(
      //               color: Colors.green[500],
      //               alignment: Alignment.center,
      //               child: Image.asset('${i.path}', fit: BoxFit.cover),
      //             ),
      //           ),
      //           Flexible(
      //             flex: 1,
      //             child: Container(
      //               color: Colors.green[300],
      //               alignment: Alignment.center,
      //               child: Text("${i.name}"),
      //             ),
      //           ),
      //         ],
      //       ),
      //     );
      //   }).toList(), // Convert to a List of Widgets after the map
      // ),

      // Ver.2
      body: Column(
        children: nameCard.map((i) {
          return Column(
            children: [
              Container(
                // color: Colors.green[500],
                alignment: Alignment.center,
                child: Image.asset(
                  '${i.path}',
                  height: 200,
                  fit: BoxFit.cover,
                ),
              ),

              Container(
                // color: Colors.green[300],
                alignment: Alignment.center,
                child: Text("${i.name}", style: TextStyle(fontSize: 18)),
              ),
            ],
          );
        }).toList(),
      ),


      floatingActionButton: FloatingActionButton.extended(
        backgroundColor: Colors.greenAccent,
        onPressed: () {
          // setState(() {
          //   size += 100.0;
          // });
        },
        icon: const Icon(Icons.add_circle_outline, size: 30, color: Color.fromARGB(255, 0, 0, 0)),
        label: const Text('Add Clothes', style: TextStyle(color: Color.fromARGB(255, 0, 0, 0))),
      ),
    );
  }
}
