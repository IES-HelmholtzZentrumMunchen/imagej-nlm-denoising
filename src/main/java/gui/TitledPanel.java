package gui;

import java.awt.*;
import java.awt.event.WindowAdapter;
import java.awt.event.WindowEvent;

/**
 * A titled panel for dialogs.
 * @author Julien Pontabry
 */
public class TitledPanel extends Panel {
    /**
     * Title of the panel.
     */
    private String m_title;

    /**
     * Insets of the panel.
     */
    private Insets insets = new Insets( 10, 10, 10, 10 );

    /**
     * Default constructor.
     */
    public TitledPanel() {
        this("");
    }

    /**
     * Constructor.
     * @param title Title of the panel
     */
    public TitledPanel(String title) {
        m_title = title;
    }

    /**
     * Insets getter.
     * @return The current insets
     */
    public Insets getInsets() {
        return insets;
    }

    /**
     * Paint the panel on the provided graphics.
     * @param g Graphics to paint on.
     */
    public void paint(Graphics g) {
        super.paint(g);
        g.setColor(this.getForeground());
        g.drawRect(5, 5, this.getWidth() - 10, this.getHeight() - 10);
        int width = g.getFontMetrics().stringWidth(m_title);
        g.setColor(this.getBackground());
        g.fillRect(10, 0, width, 10);
        g.setColor(this.getForeground());
        g.drawString(m_title, 10, 10 );
    }

    /**
     * Test main function.
     * @param args Command line arguments.
     */
    public static void main(String[] args) {
        Frame f = new Frame("TitledPanel Tester");

        TitledPanel p = new TitledPanel("Title of Panel");
        p.add(new Label("Label 1"));
        p.add(new Label("Label 2"));
        p.add(new Label("Label 3"));
        f.add(p);

        f.addWindowListener(new WindowAdapter() {
            public void windowClosing(WindowEvent e) {
                System.exit(0);
            }
        } );
        f.setBounds(300, 300, 300, 300);
        f.setVisible(true);
    }
}
