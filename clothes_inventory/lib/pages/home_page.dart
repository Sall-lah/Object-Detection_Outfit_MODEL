import 'dart:io';
import 'dart:convert';
import 'package:flutter/material.dart';
import 'package:hive_flutter/hive_flutter.dart';
import 'package:image_picker/image_picker.dart';
import 'package:flutter_dotenv/flutter_dotenv.dart';
import 'package:http/http.dart' as http;
import '../models/clothes.dart';
import 'loading_page.dart';
import 'warning_page.dart';
import 'no_detection_page.dart';

class HomePage extends StatefulWidget {
  const HomePage({super.key});
  @override
  State<HomePage> createState() => _HomePageState();
}

class _HomePageState extends State<HomePage> {
  final Box<Clothes> box = Hive.box<Clothes>('clothesBox');
  final ImagePicker picker = ImagePicker();

  // Load api from .env
  final baseUrl = dotenv.env['BASE_URL'];

  Future<void> pickImageAndSend() async {
    final XFile? picked = await picker.pickImage(source: ImageSource.gallery);
    if (picked == null) return;

    // Show loading page while waiting for response
    Navigator.of(context).push(MaterialPageRoute(
      builder: (_) => const LoadingPage(),
      fullscreenDialog: true,
    ));

    try {
      final File imgFile = File(picked.path);

      var request = http.MultipartRequest('POST', Uri.parse('$baseUrl'));
      request.files.add(await http.MultipartFile.fromPath('image', imgFile.path));

      var streamedResp = await request.send();
      final respString = await streamedResp.stream.bytesToString();

      // close loading page
      if (mounted) Navigator.of(context).pop();

      if (streamedResp.statusCode >= 200 && streamedResp.statusCode < 300) {
        final data = jsonDecode(respString);

        // If your API uses { status: 'failed', result: {name, color}, message: '...' }
        // or returns directly { name, color } we handle both.
        String name = "unknown";
        String color = "unknown";
        if (data is Map && data.containsKey('status') && data['status'] == 'failed') {
          final result = data['result'];
          if (result is Map) {
            name = result['name'] ?? 'unknown';
            color = result['color'] ?? 'unknown';
          }
          // if failed -> treat as no detection
        } else if (data is Map && data.containsKey('status') && data['status'] == 'success') {
          final result = data['result'];
          if (result is Map) {
            name = result['name'] ?? 'unknown';
            color = result['color'] ?? 'unknown';
          }
        }

        // If no object detected:
        if (name.toLowerCase() == 'unknown' || color.toLowerCase() == 'unknown') {
          // show no detection page
          Navigator.of(context).push(MaterialPageRoute(
            builder: (_) => const NoDetectionPage(),
            fullscreenDialog: true,
          ));
          return;
        }

        // check duplicates (by name+color)
        bool exists = box.values.any((c) =>
            c.name.toLowerCase() == name.toLowerCase() &&
            c.color.toLowerCase() == color.toLowerCase());

        if (exists) {
          Navigator.of(context).push(MaterialPageRoute(builder: (_) => const WarningPage()));
          return;
        }

        // Save to hive - we will keep the local path to the picked image
        await box.add(Clothes(name: name, color: color, imagePath: imgFile.path));
        setState(() {});
      } else {
        // non-2xx: show no detection or error
        Navigator.of(context).push(MaterialPageRoute(builder: (_) => const NoDetectionPage()));
      }
    } catch (e) {
      // close loading if still open
      if (Navigator.canPop(context)) Navigator.of(context).pop();
      // show error dialog
      showDialog(
        context: context,
        builder: (_) => AlertDialog(
          title: const Text('Error'),
          content: Text('Request failed: $e'),
          actions: [TextButton(onPressed: () => Navigator.pop(context), child: const Text('OK'))],
        ),
      );
    }
  }

  Future<void> deleteItem(int index) async {
    await box.deleteAt(index);
    setState(() {});
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(title: const Text('Clothes Inventory')),
      body: ValueListenableBuilder(
        valueListenable: box.listenable(),
        builder: (context, Box<Clothes> b, _) {
          if (b.isEmpty) {
            return const Center(child: Text('No clothes yet. Tap + to add.'));
          }
          return GridView.builder(
            padding: const EdgeInsets.all(8),
            itemCount: b.length,
            gridDelegate: const SliverGridDelegateWithFixedCrossAxisCount(
              crossAxisCount: 2,
              childAspectRatio: 0.7,
              mainAxisSpacing: 8,
              crossAxisSpacing: 8,
            ),
            itemBuilder: (context, i) {
              final item = b.getAt(i)!;
              return Card(
                elevation: 2,
                shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(8)),
                child: Column(
                  children: [
                    Expanded(
                      child: item.imagePath.isNotEmpty && File(item.imagePath).existsSync()
                          ? ClipRRect(
                              borderRadius: const BorderRadius.vertical(top: Radius.circular(8)),
                              child: Image.file(File(item.imagePath), fit: BoxFit.cover, width: double.infinity),
                            )
                          : Container(
                              alignment: Alignment.center,
                              child: const Text('No image'),
                            ),
                    ),
                    Padding(
                      padding: const EdgeInsets.all(8.0),
                      child: Column(
                        children: [
                          Text(item.name, style: const TextStyle(fontWeight: FontWeight.bold)),
                          const SizedBox(height: 4),
                          Text(item.color),
                          const SizedBox(height: 8),
                          Row(
                            mainAxisAlignment: MainAxisAlignment.center,
                            children: [
                              ElevatedButton.icon(
                                onPressed: () => deleteItem(i),
                                icon: const Icon(Icons.delete),
                                label: const Text('Delete'),
                                style: ElevatedButton.styleFrom(backgroundColor: Colors.red),
                              ),
                            ],
                          ),
                        ],
                      ),
                    ),
                  ],
                ),
              );
            },
          );
        },
      ),
      floatingActionButton: FloatingActionButton(
        onPressed: pickImageAndSend,
        child: const Icon(Icons.add),
      ),
    );
  }
}
