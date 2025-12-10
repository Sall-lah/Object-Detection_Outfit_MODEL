import 'package:flutter/material.dart';

class WarningPage extends StatelessWidget {
  const WarningPage({super.key});
  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(title: const Text('Duplicate Item')),
      body: Center(
        child: Column(mainAxisSize: MainAxisSize.min, children: [
          const Icon(Icons.warning, size: 80, color: Colors.orange),
          const SizedBox(height: 12),
          const Text('This clothes already exists in your inventory', textAlign: TextAlign.center),
          const SizedBox(height: 20),
          ElevatedButton(onPressed: () => Navigator.pop(context), child: const Text('Back')),
        ]),
      ),
    );
  }
}
