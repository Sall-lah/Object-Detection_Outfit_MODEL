import 'package:flutter/material.dart';

class NoDetectionPage extends StatelessWidget {
  const NoDetectionPage({super.key});
  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(title: const Text('No Detection')),
      body: Center(
        child: Column(mainAxisSize: MainAxisSize.min, children: [
          const Icon(Icons.search_off, size: 80, color: Colors.grey),
          const SizedBox(height: 12),
          const Text('No object detected in the image.', textAlign: TextAlign.center),
          const SizedBox(height: 8),
          const Text('Try a clearer photo or different angle.', textAlign: TextAlign.center),
          const SizedBox(height: 20),
          ElevatedButton(onPressed: () => Navigator.pop(context), child: const Text('Back')),
        ]),
      ),
    );
  }
}
